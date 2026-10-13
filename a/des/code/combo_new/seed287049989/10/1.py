import random

import numpy as np


BLACK = 0
GRAY = 5
COLOR_POOL = tuple(c for c in range(10) if c not in (BLACK, GRAY))
GEN_ATTEMPTS = 500
RECT_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_rect_frame(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(sub, expected)


def _extract_legend(grid: np.ndarray) -> tuple[str, list[int]]:
    bottom_idx = np.flatnonzero((grid[-1] != BLACK) & (grid[-1] != GRAY))
    right_idx = np.flatnonzero((grid[:, -1] != BLACK) & (grid[:, -1] != GRAY))

    if bottom_idx.size and not right_idx.size:
        if bottom_idx.size > 1:
            assert np.all(np.diff(bottom_idx) == 1), "bottom legend must be contiguous"
        return "bottom", [int(grid[-1, c]) for c in bottom_idx]
    if right_idx.size and not bottom_idx.size:
        if right_idx.size > 1:
            assert np.all(np.diff(right_idx) == 1), "right legend must be contiguous"
        return "right", [int(grid[r, -1]) for r in right_idx]

    raise AssertionError("expected exactly one legend on the bottom edge or right edge")


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray_mask = grid == GRAY
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(gray_mask):
        box = _bbox(comp)
        assert _is_rect_frame(gray_mask, box), "all gray components must be rectangular frames"
        frames.append(box)
    assert frames, "at least one gray frame is required"
    return sorted(frames)


def _interior_color_count(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = set(map(int, np.unique(interior))) - {BLACK, GRAY}
    return len(colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rectangular border from the legend index given by its interior color diversity.

    Input structure:
    - The background is black.
    - Several one-pixel-thick gray rectangular frames enclose black interiors containing scattered non-black, non-gray colored cells.
    - A one-cell-thick legend sits on the bottom edge or right edge as a left-to-right or top-to-bottom sequence of distinct single-pixel colors.

    Rule:
    - Read the legend colors in order and remove the legend by turning its whole row or column black.
    - For each gray rectangular frame, count how many distinct non-black colors appear strictly inside the rectangle.
    - Repaint the entire gray border with the legend color at that 1-based index while leaving the interior unchanged.

    Key insight:
    - The colored dots inside a frame matter only through their number of distinct colors, and the edge strip is an ordered lookup table rather than a direct color match.
    """
    g = np.asarray(grid, dtype=int)
    orientation, legend = _extract_legend(g)
    assert legend and len(legend) == len(set(legend)), "legend colors must be distinct"

    out = g.copy()
    if orientation == "bottom":
        out[-1, :] = BLACK
    else:
        out[:, -1] = BLACK

    for box in _find_frames(g):
        n_colors = _interior_color_count(g, box)
        assert 1 <= n_colors <= len(legend), "interior color count must index the legend"
        _paint_frame(out, box, legend[n_colors - 1])

    return out


def _sample_counts(num_rects: int, legend_len: int) -> list[int]:
    while True:
        counts = [random.randint(1, legend_len) for _ in range(num_rects)]
        if num_rects == 1 or len(set(counts)) >= 2:
            return counts


def _place_rectangles(
    grid: np.ndarray, orientation: str, counts: list[int]
) -> list[tuple[tuple[int, int, int, int], int]] | None:
    h, w = grid.shape
    occupied = np.zeros_like(grid, dtype=bool)
    rect_specs: list[tuple[tuple[int, int, int, int], int]] = []

    max_bottom = h - (3 if orientation == "bottom" else 2)
    max_right = w - (3 if orientation == "right" else 2)

    for count in sorted(counts, reverse=True):
        placed = False
        for _ in range(RECT_ATTEMPTS):
            max_h = min(10, max_bottom)
            max_w = min(11, max_right)
            if max_h < 4 or max_w < 4:
                return None

            rect_h = random.randint(4, max_h)
            rect_w = random.randint(4, max_w)
            if (rect_h - 2) * (rect_w - 2) < count:
                continue

            r0_max = max_bottom - rect_h + 1
            c0_max = max_right - rect_w + 1
            if r0_max < 1 or c0_max < 1:
                continue

            r0 = random.randint(1, r0_max)
            c0 = random.randint(1, c0_max)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r1 + 2)
            cc1 = min(w, c1 + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            _paint_frame(grid, (r0, c0, r1, c1), GRAY)
            interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            chosen_colors = random.sample(COLOR_POOL, count)
            extra_cap = min(len(interior), count + max(1, len(interior) // 3))
            dot_count = random.randint(count, extra_cap)
            dot_cells = random.sample(interior, dot_count)
            for (r, c), color in zip(dot_cells[:count], chosen_colors):
                grid[r, c] = color
            for r, c in dot_cells[count:]:
                grid[r, c] = random.choice(chosen_colors)

            occupied[rr0:rr1, cc0:cc1] = True
            rect_specs.append(((r0, c0, r1, c1), count))
            placed = True
            break
        if not placed:
            return None

    return rect_specs


def _place_legend(grid: np.ndarray, orientation: str, legend: list[int]) -> None:
    h, w = grid.shape
    if orientation == "bottom":
        start = random.randint(0, w - len(legend) - 1)
        grid[-1, start : start + len(legend)] = legend
    else:
        start = random.randint(0, h - len(legend) - 1)
        grid[start : start + len(legend), -1] = legend


def _oracle_output(
    grid: np.ndarray,
    orientation: str,
    legend: list[int],
    rect_specs: list[tuple[tuple[int, int, int, int], int]],
) -> np.ndarray:
    out = grid.copy()
    if orientation == "bottom":
        out[-1, :] = BLACK
    else:
        out[:, -1] = BLACK
    for bbox, count in rect_specs:
        _paint_frame(out, bbox, legend[count - 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("bottom", "right"))
    height = random.randint(12, 28)
    width = random.randint(12, 28)
    max_legend = min(6, len(COLOR_POOL), (width - 1) if orientation == "bottom" else (height - 1))
    if max_legend < 2:
        return None

    max_rects = 4 if min(height, width) >= 18 else 3
    num_rects = random.randint(2, max_rects)
    legend_len = random.randint(2, max_legend)
    counts = _sample_counts(num_rects, legend_len)
    legend = random.sample(COLOR_POOL, legend_len)

    grid = np.full((height, width), BLACK, dtype=int)
    rect_specs = _place_rectangles(grid, orientation, counts)
    if rect_specs is None:
        return None

    _place_legend(grid, orientation, legend)

    expected = _oracle_output(grid, orientation, legend, rect_specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
