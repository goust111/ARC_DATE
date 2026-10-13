from __future__ import annotations

import random

import numpy as np


ArrowKey = tuple[tuple[int, int], ...]
FrameJob = tuple[np.ndarray, ArrowKey]


ARROW_PATTERNS: dict[str, ArrowKey] = {
    "up": ((0, 1), (1, 0), (1, 1)),
    "right": ((0, 0), (1, 0), (1, 1)),
    "down": ((0, 0), (0, 1), (1, 0)),
    "left": ((0, 0), (0, 1), (1, 1)),
}
ALL_DIRECTIONS = tuple(ARROW_PATTERNS)
LEGEND_ROW_STEP = 3
LEGEND_HEIGHT = 11
LEGEND_WIDTH = 3
FRAME_SIZES = (6, 8, 10)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalized_arrow(coords: np.ndarray) -> ArrowKey | None:
    if int(coords.shape[0]) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None
    normalized = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return normalized if normalized in ARROW_PATTERNS.values() else None


def _non_bg_components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    comps: list[tuple[int, np.ndarray]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components(grid == color):
            comps.append((color, coords))
    return comps


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[ArrowKey, int]:
    comps = _non_bg_components(grid, bg)
    singleton_lookup: dict[tuple[int, int], int] = {}
    for color, coords in comps:
        if int(coords.shape[0]) == 1:
            singleton_lookup[(int(coords[0, 0]), int(coords[0, 1]))] = int(color)

    mapping: dict[ArrowKey, int] = {}
    h, w = grid.shape
    for _color, coords in comps:
        arrow = _normalized_arrow(coords)
        if arrow is None:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if c1 + 1 >= w:
            continue
        candidates = [(r, c1 + 1) for r in range(r0, r1 + 1) if (r, c1 + 1) in singleton_lookup]
        if len(candidates) != 1:
            continue
        palette_color = int(singleton_lookup[candidates[0]])
        previous = mapping.get(arrow)
        assert previous is None or previous == palette_color, "legend direction must map to one color"
        mapping[arrow] = palette_color

    assert len(mapping) == 4, "expected a four-entry direction legend"
    return mapping


def _is_square_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 4:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(grid[r0 : r1 + 1, c0 : c1 + 1] == color, border)


def _frame_arrow(grid: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int, bg: int) -> ArrowKey:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    n = interior.shape[0]
    expected_top = (n - 2) // 2
    candidates: list[ArrowKey] = []

    for color0 in np.unique(interior):
        color = int(color0)
        if color in (bg, frame_color):
            continue
        for coords in _components(interior == color):
            arrow = _normalized_arrow(coords)
            if arrow is None:
                continue
            rr0, cc0, rr1, cc1 = _bbox(coords)
            if (rr0, cc0, rr1, cc1) == (expected_top, expected_top, expected_top + 1, expected_top + 1):
                candidates.append(arrow)

    assert len(candidates) == 1, "each frame must contain one centered L-arrow"
    return candidates[0]


def _frame_jobs(grid: np.ndarray, bg: int) -> list[FrameJob]:
    jobs: list[FrameJob] = []
    for color, coords in _non_bg_components(grid, bg):
        if not _is_square_frame(grid, color, coords):
            continue
        arrow = _frame_arrow(grid, _bbox(coords), color, bg)
        jobs.append((coords, arrow))
    assert jobs, "expected at least one square frame"
    return jobs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow square frame by reading the direction-to-color legend formed by tiny L-arrows.

    Input structure:
    - The background is the most common color.
    - Several same-style hollow square frames are scattered in the grid, each drawn as a 1-pixel-thick border.
    - Each frame contains one centered 3-pixel L-shaped arrow, and one corner region contains four legend rows:
      each row has the same arrow style plus a single color pixel immediately to its right.

    Rule:
    - Read the legend as a mapping from arrow orientation to repaint color.
    - For every square frame, identify the orientation of its centered L-arrow.
    - Recolor the whole frame border to the legend color for that orientation, while leaving the inner arrow untouched.
    - Leave the legend and every non-frame pixel unchanged.

    Key insight:
    - The tiny rotated L-shapes are not decoration; they are lookup keys that connect each frame to one legend color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    legend = _legend_mapping(g, bg)
    jobs = _frame_jobs(g, bg)

    out = g.copy()
    for coords, arrow in jobs:
        out[coords[:, 0], coords[:, 1]] = int(legend[arrow])
    return out


def _draw_arrow(grid: np.ndarray, top: int, left: int, color: int, direction: str) -> None:
    for dr, dc in ARROW_PATTERNS[direction]:
        grid[top + dr, left + dc] = int(color)


def _draw_frame(grid: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    grid[top, left : left + size] = int(color)
    grid[top + size - 1, left : left + size] = int(color)
    grid[top : top + size, left] = int(color)
    grid[top : top + size, left + size - 1] = int(color)


def _paint_frame_arrow(grid: np.ndarray, top: int, left: int, size: int, arrow_color: int, direction: str) -> None:
    inner = size - 2
    start = top + 1 + (inner - 2) // 2
    _draw_arrow(grid, start, left + 1 + (inner - 2) // 2, arrow_color, direction)


def _legend_origin(h: int, w: int, corner: str) -> tuple[int, int]:
    row_pad = random.randint(0, min(2, h - LEGEND_HEIGHT))
    col_pad = random.randint(0, min(2, w - LEGEND_WIDTH))
    top = row_pad if corner[0] == "t" else h - LEGEND_HEIGHT - row_pad
    left = col_pad if corner[1] == "l" else w - LEGEND_WIDTH - col_pad
    return int(top), int(left)


def _reserve(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    mask[r0:r1, c0:c1] = True


def _can_place(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(mask[r0:r1, c0:c1].any())


def _frame_count_for_area(area: int) -> int:
    if area < 520:
        return random.randint(4, 5)
    if area < 700:
        return random.randint(4, 6)
    return random.randint(5, 7)


def _size_choices(h: int, w: int, frame_count: int) -> tuple[int, ...]:
    usable = min(h, w)
    sizes = [size for size in FRAME_SIZES if size + 2 <= usable]
    if frame_count >= 6 and 10 in sizes:
        sizes.remove(10)
    if not sizes:
        sizes = [6]
    return tuple(sizes)


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    arrow_color: int,
    legend_colors: dict[str, int],
) -> None:
    corner = random.choice(("tl", "tr", "bl", "br"))
    top, left = _legend_origin(*grid.shape, corner)
    directions = list(ALL_DIRECTIONS)
    random.shuffle(directions)
    for idx, direction in enumerate(directions):
        row = top + idx * LEGEND_ROW_STEP
        _draw_arrow(grid, row, left, arrow_color, direction)
        grid[row + random.randint(0, 1), left + 2] = int(legend_colors[direction])
    _reserve(occupied, top, left, LEGEND_HEIGHT, LEGEND_WIDTH, pad=1)


def _place_frames(
    grid: np.ndarray,
    occupied: np.ndarray,
    frame_color: int,
    arrow_color: int,
    legend_colors: dict[str, int],
) -> list[tuple[np.ndarray, int]]:
    h, w = grid.shape
    frame_count = _frame_count_for_area(int(h * w))
    directions = list(ALL_DIRECTIONS) + [random.choice(ALL_DIRECTIONS) for _ in range(frame_count - 4)]
    random.shuffle(directions)
    sizes = _size_choices(h, w, frame_count)
    placed: list[tuple[np.ndarray, int]] = []

    for direction in directions:
        size = random.choice(sizes)
        found = False
        row_choices = list(range(1, h - size))
        col_choices = list(range(1, w - size))
        random.shuffle(row_choices)
        random.shuffle(col_choices)
        for top in row_choices:
            for left in col_choices:
                if not _can_place(occupied, top, left, size, size, pad=1):
                    continue
                _draw_frame(grid, top, left, size, frame_color)
                _paint_frame_arrow(grid, top, left, size, arrow_color, direction)
                _reserve(occupied, top, left, size, size, pad=1)
                coords = np.argwhere(
                    (np.indices((h, w))[0] >= top)
                    & (np.indices((h, w))[0] < top + size)
                    & (np.indices((h, w))[1] >= left)
                    & (np.indices((h, w))[1] < left + size)
                    & (
                        (np.indices((h, w))[0] == top)
                        | (np.indices((h, w))[0] == top + size - 1)
                        | (np.indices((h, w))[1] == left)
                        | (np.indices((h, w))[1] == left + size - 1)
                    )
                )
                placed.append((coords, legend_colors[direction]))
                found = True
                break
            if found:
                break
        if not found:
            raise RuntimeError("failed to place all frames")

    return placed


def _expected_output(inp: np.ndarray, frame_jobs: list[tuple[np.ndarray, int]]) -> np.ndarray:
    out = inp.copy()
    for coords, color in frame_jobs:
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    frame_color = int(palette[1])
    arrow_color = int(palette[2])
    legend_color_values = [int(c) for c in palette[3:7]]
    random.shuffle(legend_color_values)
    legend_colors = dict(zip(ALL_DIRECTIONS, legend_color_values))

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    try:
        _place_legend(grid, occupied, arrow_color, legend_colors)
        frame_jobs = _place_frames(grid, occupied, frame_color, arrow_color, legend_colors)
    except RuntimeError:
        return None

    expected = _expected_output(grid, frame_jobs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-and-legend puzzle")
