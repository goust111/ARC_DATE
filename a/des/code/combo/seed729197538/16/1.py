from __future__ import annotations

import random

import numpy as np


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_left_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    col = grid[:, 0]
    mapping: dict[int, int] = {}
    r = 0
    while r < len(col):
        if int(col[r]) == bg:
            r += 1
            continue
        start = r
        while r < len(col) and int(col[r]) != bg:
            r += 1
        if r - start == 2:
            key = int(col[start])
            value = int(col[start + 1])
            if key in mapping:
                assert mapping[key] == value, "legend key must map uniquely"
            mapping[key] = value
    assert mapping, "expected at least one left-column legend entry"
    return mapping


def _is_rect_frame_component(
    grid: np.ndarray, color: int, points: list[tuple[int, int]]
) -> tuple[bool, tuple[int, int, int, int] | None]:
    r0, c0, r1, c1 = _bbox(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False, None
    if len(points) != 2 * (height + width) - 4:
        return False, None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    expected = np.zeros((height, width), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(patch, expected):
        return False, None
    return True, (int(r0), int(c0), int(r1), int(c1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each rectangle's interior dot color using the left-column legend.

    The legend lives in column 0 as isolated vertical pairs:
      top pixel   = dot color
      bottom pixel = replacement color

    Each colored rectangular frame encloses background plus one dot color.
    Only those dot pixels are recolored; frames, legend, and background stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _majority_color(g)
    legend = _parse_left_legend(g, bg)
    out = g.copy()

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color):
            is_frame, box = _is_rect_frame_component(g, color, comp)
            if not is_frame or box is None:
                continue
            r0, c0, r1, c1 = box
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            dot_mask = inner != bg
            if not np.any(dot_mask):
                continue
            dot_colors = np.unique(inner[dot_mask])
            assert len(dot_colors) == 1, "each frame interior must contain a single dot color"
            dot_color = int(dot_colors[0])
            assert dot_color in legend, "dot color must exist in the legend"
            fill_color = legend[dot_color]
            inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
            inner_out[dot_mask] = fill_color

    return out


GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (14, 26)
FRAME_COUNT_CHOICES = (1, 2, 3)
EXTRA_LEGEND_PROB = 0.45
FRAME_SIZE_MIN = 3
FRAME_SIZE_MAX = 9
FRAME_GAP = 1
GEN_RETRIES = 500
FRAME_PLACE_RETRIES = 200


def _legend_rows(height: int, count: int, rng: random.Random) -> list[int]:
    required = 2 * count + max(0, count - 1)
    assert required <= height
    slack = height - required
    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[rng.randrange(count + 1)] += 1

    rows: list[int] = []
    r = extras[0]
    for i in range(count):
        rows.append(r)
        r += 2
        if i + 1 < count:
            r += 1 + extras[i + 1]
    return rows


def _sample_isolated_specks(
    interior_h: int, interior_w: int, rng: random.Random
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(interior_h) for c in range(interior_w)]
    rng.shuffle(cells)

    available: list[tuple[int, int]] = []
    for r, c in cells:
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in available):
            available.append((r, c))

    assert available
    max_count = min(len(available), max(1, min(6, (interior_h * interior_w + 1) // 3)))
    count = rng.randint(1, max_count)
    rng.shuffle(available)
    return available[:count]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _background_is_majority(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    best = int(values[np.argmax(counts)])
    return best == bg


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    max_entries = (9 - frame_count) // 2
    legend_count = frame_count
    if legend_count < max_entries and rng.random() < EXTRA_LEGEND_PROB:
        legend_count += 1

    min_h = max(GRID_HEIGHT_RANGE[0], 3 * legend_count + 1)
    h = rng.randint(min_h, GRID_HEIGHT_RANGE[1])
    w = rng.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])

    color_count = 1 + frame_count + 2 * legend_count
    colors = rng.sample(list(range(10)), color_count)
    rng.shuffle(colors)
    bg = colors[0]
    dot_colors = colors[1 : 1 + legend_count]
    fill_colors = colors[1 + legend_count : 1 + 2 * legend_count]
    border_colors = colors[1 + 2 * legend_count :]

    grid = np.full((h, w), bg, dtype=int)
    out = grid.copy()

    legend_positions = _legend_rows(h, legend_count, rng)
    for row, dot_color, fill_color in zip(
        legend_positions, dot_colors, fill_colors, strict=False
    ):
        grid[row, 0] = dot_color
        grid[row + 1, 0] = fill_color
        out[row, 0] = dot_color
        out[row + 1, 0] = fill_color

    used_key_indices = rng.sample(range(legend_count), frame_count)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, :2] = True

    for frame_index, border_color in enumerate(border_colors):
        dot_color = dot_colors[used_key_indices[frame_index]]
        fill_color = fill_colors[used_key_indices[frame_index]]
        placed = False

        for _ in range(FRAME_PLACE_RETRIES):
            frame_h = rng.randint(FRAME_SIZE_MIN, min(FRAME_SIZE_MAX, h - 2))
            frame_w = rng.randint(FRAME_SIZE_MIN, min(FRAME_SIZE_MAX, w - 4))
            r0 = rng.randint(1, h - frame_h - 1)
            c0 = rng.randint(3, w - frame_w - 1)
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1

            pad_r0 = max(0, r0 - FRAME_GAP)
            pad_r1 = min(h, r1 + FRAME_GAP + 1)
            pad_c0 = max(0, c0 - FRAME_GAP)
            pad_c1 = min(w, c1 + FRAME_GAP + 1)
            if np.any(blocked[pad_r0:pad_r1, pad_c0:pad_c1]):
                continue

            interior_h = frame_h - 2
            interior_w = frame_w - 2
            specks = _sample_isolated_specks(interior_h, interior_w, rng)

            _draw_frame(grid, r0, c0, r1, c1, border_color)
            _draw_frame(out, r0, c0, r1, c1, border_color)
            for sr, sc in specks:
                rr = r0 + 1 + sr
                cc = c0 + 1 + sc
                grid[rr, cc] = dot_color
                out[rr, cc] = fill_color

            blocked[pad_r0:pad_r1, pad_c0:pad_c1] = True
            placed = True
            break

        if not placed:
            raise RuntimeError("failed to place all frames")

    if not _background_is_majority(grid, bg):
        raise RuntimeError("background must remain the dominant color")
    return grid, out


def generate_input() -> np.ndarray:
    rng = random
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _build_case(rng)
        except RuntimeError:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
