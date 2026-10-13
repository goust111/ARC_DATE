from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
GREY = 5
MAX_GENERATION_ATTEMPTS = 400
SIDES = ("top", "bottom", "left", "right")


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _gap_width(line: np.ndarray, color: int, bg: int) -> int | None:
    vals = np.asarray(line, dtype=int)
    if vals.size < 4:
        return None
    if int(vals[0]) != color or int(vals[-1]) != color:
        return None

    interior = vals[1:-1]
    miss = interior != color
    if not miss.any():
        return None
    miss_idx = np.flatnonzero(miss)
    if miss_idx.size < 2:
        return None
    if int(miss_idx[-1]) - int(miss_idx[0]) + 1 != int(miss_idx.size):
        return None
    if not np.all(interior[miss] == bg):
        return None
    if not np.all(interior[~miss] == color):
        return None
    return int(miss_idx.size)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], str]:
    candidates: list[tuple[int, tuple[int, int, int, int], str]] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        mask = grid == int(color)
        b = _bbox(mask)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if bool(mask[r0 + 1 : r1, c0 + 1 : c1].any()):
            continue

        top = _gap_width(grid[r0, c0 : c1 + 1], color, bg)
        bottom = _gap_width(grid[r1, c0 : c1 + 1], color, bg)
        left = _gap_width(grid[r0 : r1 + 1, c0], color, bg)
        right = _gap_width(grid[r0 : r1 + 1, c1], color, bg)
        gaps = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        }
        gap_sides = [side for side, width in gaps.items() if width is not None]
        if len(gap_sides) != 1:
            continue

        if top is None and not np.all(grid[r0, c0 : c1 + 1] == color):
            continue
        if bottom is None and not np.all(grid[r1, c0 : c1 + 1] == color):
            continue
        if left is None and not np.all(grid[r0 : r1 + 1, c0] == color):
            continue
        if right is None and not np.all(grid[r0 : r1 + 1, c1] == color):
            continue

        border_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        gap_side = gap_sides[0]
        if int(mask.sum()) != border_area - int(gaps[gap_side]):
            continue
        candidates.append((int(color), b, gap_side))

    assert len(candidates) == 1, "expected one gapped rectangular frame"
    return candidates[0]


def _find_pattern_and_marker(
    inner: np.ndarray,
    *,
    bg: int,
    frame_color: int,
) -> tuple[np.ndarray, tuple[int, int], int]:
    payload_colors = sorted(set(map(int, inner.ravel())) - {int(bg), int(frame_color)})
    assert len(payload_colors) == 2, "expected one pattern color and one marker color"

    counts = {color: int(np.count_nonzero(inner == color)) for color in payload_colors}
    if GREY in counts:
        pattern_color = GREY
        marker_color = next(color for color in payload_colors if color != GREY)
    else:
        marker_candidates = [color for color, count in counts.items() if count == 1]
        assert len(marker_candidates) == 1, "fallback pattern inference requires a unique singleton marker"
        marker_color = marker_candidates[0]
        pattern_color = next(color for color in payload_colors if color != marker_color)

    pattern_mask = inner == int(pattern_color)
    assert 1 <= int(pattern_mask.sum()) <= 8
    assert _is_single_component(pattern_mask), "pattern must be one 4-connected component"

    marker_cells = np.argwhere(inner == int(marker_color))
    assert marker_cells.shape == (1, 2), "marker must be a unique singleton"
    marker = tuple(map(int, marker_cells[0]))
    return pattern_mask, marker, int(marker_color)


def _mirrored_coords(coords: np.ndarray, shape: tuple[int, int], gap_side: str) -> np.ndarray:
    mirrored = coords.copy()
    if gap_side in {"top", "bottom"}:
        mirrored[:, 0] = int(shape[0]) - 1 - mirrored[:, 0]
    else:
        mirrored[:, 1] = int(shape[1]) - 1 - mirrored[:, 1]
    return mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the inner grey pattern across the axis encoded by the frame gap, then recolor it with the marker color.

    Input structure:
    - The background is black.
    - One non-background color forms a one-pixel rectangular frame whose top, bottom, left, or right edge has exactly one continuous gap of width at least two.
    - Inside the frame there is one 4-connected grey pattern with at most eight cells, plus exactly one isolated singleton marker pixel of a different color.

    Rule:
    - Detect which frame edge contains the gap.
    - If the gap is on the top or bottom edge, mirror the grey pattern across the horizontal midline of the frame interior; if the gap is on the left or right edge, mirror it across the vertical midline.
    - Recolor the mirrored pattern with the singleton marker's color, then erase the original grey pattern and the marker while leaving the frame unchanged.

    Key insight:
    - The missing border segment encodes the mirror direction, while the lone interior marker contributes only the final paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BACKGROUND if bool(np.any(g == BACKGROUND)) else _most_common_color(g)
    frame_color, (r0, c0, r1, c1), gap_side = _find_frame(g, bg)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    pattern_mask, marker, marker_color = _find_pattern_and_marker(
        inner,
        bg=bg,
        frame_color=frame_color,
    )

    coords = np.argwhere(pattern_mask)
    mirrored = _mirrored_coords(coords, inner.shape, gap_side)

    out = g.copy()
    inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
    inner_out[pattern_mask] = bg
    out[r0 + 1 + marker[0], c0 + 1 + marker[1]] = bg
    out[r0 + 1 + mirrored[:, 0], c0 + 1 + mirrored[:, 1]] = marker_color
    return out


def _grow_connected_shape(height: int, width: int, area: int) -> np.ndarray | None:
    if area > height * width:
        return None
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], height, width))
    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors4(cell[0], cell[1], height, width):
            if nbr not in cells:
                frontier.add(nbr)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        shape[r - r0, c - c0] = True
    return shape


def _axis_asymmetric(shape: np.ndarray, gap_side: str) -> bool:
    coords = np.argwhere(shape)
    cells = {tuple(map(int, coord)) for coord in coords}
    h, w = shape.shape
    if gap_side in {"top", "bottom"}:
        mirrored = {(h - 1 - r, c) for r, c in cells}
    else:
        mirrored = {(r, w - 1 - c) for r, c in cells}
    return cells != mirrored


def _candidate_pattern_rows(inner_h: int, ph: int, gap_side: str) -> list[int]:
    if gap_side in {"top", "bottom"}:
        upper = list(range(0, max(0, inner_h // 2 - ph + 1)))
        lower_start = (inner_h + 1) // 2
        lower = list(range(lower_start, inner_h - ph + 1))
        return upper + lower
    return list(range(0, inner_h - ph + 1))


def _candidate_pattern_cols(inner_w: int, pw: int, gap_side: str) -> list[int]:
    if gap_side in {"left", "right"}:
        left = list(range(0, max(0, inner_w // 2 - pw + 1)))
        right_start = (inner_w + 1) // 2
        right = list(range(right_start, inner_w - pw + 1))
        return left + right
    return list(range(0, inner_w - pw + 1))


def _place_pattern(inner_h: int, inner_w: int, gap_side: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    max_ph = min(4, inner_h)
    max_pw = min(4, inner_w)
    dims = [
        (ph, pw)
        for ph in range(2, max_ph + 1)
        for pw in range(2, max_pw + 1)
        if ph * pw >= 3
    ]
    random.shuffle(dims)
    for ph, pw in dims:
        max_area = min(8, ph * pw)
        areas = list(range(3, max_area + 1))
        random.shuffle(areas)
        for area in areas:
            for _ in range(40):
                shape = _grow_connected_shape(ph, pw, area)
                if shape is None or not _axis_asymmetric(shape, gap_side):
                    continue
                row_choices = _candidate_pattern_rows(inner_h, ph, gap_side)
                col_choices = _candidate_pattern_cols(inner_w, pw, gap_side)
                if not row_choices or not col_choices:
                    continue
                random.shuffle(row_choices)
                random.shuffle(col_choices)
                local = {tuple(map(int, coord)) for coord in np.argwhere(shape)}
                for top in row_choices:
                    for left in col_choices:
                        placed = {(top + r, left + c) for r, c in local}
                        coords = np.array(sorted(placed), dtype=int)
                        mirrored = _mirrored_coords(coords, (inner_h, inner_w), gap_side)
                        mirrored_set = {tuple(map(int, coord)) for coord in mirrored}
                        if placed == mirrored_set:
                            continue
                        if placed & mirrored_set:
                            continue
                        return placed, mirrored_set
    return None


def _marker_candidates(
    inner_h: int,
    inner_w: int,
    pattern: set[tuple[int, int]],
    mirrored: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    occupied = set(pattern) | set(mirrored)
    candidates: list[tuple[int, int]] = []
    for r in range(1, inner_h - 1):
        for c in range(1, inner_w - 1):
            if (r, c) in occupied:
                continue
            if any((nr, nc) in pattern for nr, nc in _neighbors4(r, c, inner_h, inner_w)):
                continue
            candidates.append((r, c))
    return candidates


def _make_expected(
    inp: np.ndarray,
    *,
    frame_top: int,
    frame_left: int,
    pattern: set[tuple[int, int]],
    mirrored: set[tuple[int, int]],
    marker: tuple[int, int],
    marker_color: int,
) -> np.ndarray:
    out = inp.copy()
    for r, c in pattern:
        out[frame_top + 1 + r, frame_left + 1 + c] = BACKGROUND
    out[frame_top + 1 + marker[0], frame_left + 1 + marker[1]] = BACKGROUND
    for r, c in mirrored:
        out[frame_top + 1 + r, frame_left + 1 + c] = marker_color
    return out


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(6, 14)
    inner_w = random.randint(6, 14)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    max_h = min(30, outer_h + 10)
    max_w = min(30, outer_w + 10)
    grid_h = random.randint(outer_h + 2, max_h)
    grid_w = random.randint(outer_w + 2, max_w)

    frame_top = random.randint(1, grid_h - outer_h - 1)
    frame_left = random.randint(1, grid_w - outer_w - 1)

    colors = [c for c in range(1, 10) if c != GREY]
    frame_color, marker_color = random.sample(colors, 2)
    gap_side = random.choice(SIDES)

    pattern_data = _place_pattern(inner_h, inner_w, gap_side)
    if pattern_data is None:
        return None
    pattern, mirrored = pattern_data

    marker_choices = _marker_candidates(inner_h, inner_w, pattern, mirrored)
    if not marker_choices:
        return None
    marker = random.choice(marker_choices)

    inp = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    r0, c0 = frame_top, frame_left
    r1, c1 = frame_top + outer_h - 1, frame_left + outer_w - 1
    inp[r0, c0 : c1 + 1] = frame_color
    inp[r1, c0 : c1 + 1] = frame_color
    inp[r0 : r1 + 1, c0] = frame_color
    inp[r0 : r1 + 1, c1] = frame_color

    if gap_side in {"top", "bottom"}:
        gap_len = random.randint(2, inner_w)
        start = random.randint(0, inner_w - gap_len)
        rr = r0 if gap_side == "top" else r1
        inp[rr, c0 + 1 + start : c0 + 1 + start + gap_len] = BACKGROUND
    else:
        gap_len = random.randint(2, inner_h)
        start = random.randint(0, inner_h - gap_len)
        cc = c0 if gap_side == "left" else c1
        inp[r0 + 1 + start : r0 + 1 + start + gap_len, cc] = BACKGROUND

    for r, c in pattern:
        inp[r0 + 1 + r, c0 + 1 + c] = GREY
    inp[r0 + 1 + marker[0], c0 + 1 + marker[1]] = marker_color

    expected = _make_expected(
        inp,
        frame_top=frame_top,
        frame_left=frame_left,
        pattern=pattern,
        mirrored=mirrored,
        marker=marker,
        marker_color=marker_color,
    )
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-frame ARC input")
