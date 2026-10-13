from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(cells, dtype=int))

    return comps


def _frame_bbox_from_component(grid: np.ndarray, coords: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    expected = 2 * height + 2 * width - 4
    if int(coords.shape[0]) != int(expected):
        return None

    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not bool(np.array_equal(local, border)):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if bool(np.any(interior == int(color))):
        return None
    return r0, c0, r1, c1


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _mark_frame(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True


def _internal_marker_count(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int, frame_color: int) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    markers = (interior != int(bg)) & (interior != int(frame_color))
    return int(markers.sum())


def _detect_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components(grid == int(color)):
            bbox = _frame_bbox_from_component(grid, coords, color)
            if bbox is not None:
                candidates.append((color, bbox))

    assert candidates, "expected at least one closed rectangular frame"
    by_color: dict[int, int] = {}
    for color, bbox in candidates:
        r0, c0, r1, c1 = bbox
        by_color[color] = by_color.get(color, 0) + 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4

    frame_color = max(by_color, key=by_color.get)
    frames = [bbox for color, bbox in candidates if int(color) == int(frame_color)]
    assert frames, "expected frames sharing one visible border color"
    return int(frame_color), frames


def _key_colors(grid: np.ndarray, bg: int, frames: list[tuple[int, int, int, int]]) -> list[int]:
    frame_mask = np.zeros(grid.shape, dtype=bool)
    interior_mask = np.zeros(grid.shape, dtype=bool)
    for bbox in frames:
        r0, c0, r1, c1 = map(int, bbox)
        _mark_frame(frame_mask, bbox)
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    key_mask = (grid != int(bg)) & ~frame_mask & ~interior_mask
    coords = np.argwhere(key_mask)
    assert coords.size, "expected an external key-pixel sequence"

    rows = {int(r) for r, _c in coords}
    cols = {int(c) for _r, c in coords}
    if len(rows) == 1:
        ordered = sorted((int(r), int(c)) for r, c in coords)
    else:
        assert len(cols) == 1, "key pixels must form one row or one column"
        ordered = sorted((int(r), int(c)) for r, c in coords)

    colors = [int(grid[r, c]) for r, c in ordered]
    assert len(colors) == len(set(colors)), "key pixel colors must be distinct"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed rectangle frame by indexing into an external key-pixel sequence with its interior marker count.

    Input structure:
    - The most common color is the background.
    - One non-background color forms several one-pixel-thick closed rectangular
      frames; each frame contains isolated single-pixel marker colors inside it.
    - Outside the frames, a single row or single column of distinct key pixels is
      read in spatial order, with positions numbered from 1.

    Rule:
    - For each rectangular frame, count the non-background, non-frame-color pixels
      strictly inside the frame.
    - Use that count N as a one-based index into the key sequence and take the
      Nth key color.
    - Recolor all pixels of that frame to the selected key color while preserving
      the internal markers, background, and key pixels.

    Key insight:
    - The number of enclosed marker pixels is not the output value itself; it is
      an index into the visible key sequence, and the indexed color is applied
      back to the enclosing frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frame_color, frames = _detect_frames(g, bg)
    keys = _key_colors(g, bg, frames)

    out = g.copy()
    for bbox in frames:
        count = _internal_marker_count(g, bbox, bg, frame_color)
        assert 1 <= count <= len(keys), "interior marker count must index a key pixel"
        _paint_frame(out, bbox, keys[count - 1])
    return out


# --------------- generator ---------------

COLOR_COUNT = 10
GENERATION_ATTEMPTS = 500
GRID_SIZE_RANGE = (14, 30)
FRAME_COUNT_RANGE = (2, 5)
KEY_COUNT_RANGE = (3, 6)
FRAME_SIDE_RANGE = (4, 8)
FRAME_GAP = 1


def _dominant_bg(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = int(counts.get(int(bg), 0))
    return all(bg_count > int(v) for color, v in counts.items() if int(color) != int(bg))


def _marker_capacity(outer_h: int, outer_w: int) -> int:
    inner_h = int(outer_h - 2)
    inner_w = int(outer_w - 2)
    return int(((inner_h + 1) // 2) * ((inner_w + 1) // 2))


def _sample_frame_size(marker_count: int) -> tuple[int, int] | None:
    for _ in range(100):
        outer_h = random.randint(*FRAME_SIDE_RANGE)
        outer_w = random.randint(*FRAME_SIDE_RANGE)
        if _marker_capacity(outer_h, outer_w) >= int(marker_count):
            return int(outer_h), int(outer_w)
    return None


def _can_place(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = map(int, blocked.shape)
    r0, c0, r1, c1 = map(int, bbox)
    pr0 = max(0, r0 - FRAME_GAP)
    pc0 = max(0, c0 - FRAME_GAP)
    pr1 = min(h - 1, r1 + FRAME_GAP)
    pc1 = min(w - 1, c1 + FRAME_GAP)
    return not bool(blocked[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _reserve(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = map(int, blocked.shape)
    r0, c0, r1, c1 = map(int, bbox)
    pr0 = max(0, r0 - FRAME_GAP)
    pc0 = max(0, c0 - FRAME_GAP)
    pr1 = min(h - 1, r1 + FRAME_GAP)
    pc1 = min(w - 1, c1 + FRAME_GAP)
    blocked[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _sample_spaced_interior_cells(bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = map(int, bbox)
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for _ in range(200):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == int(count):
                    return chosen
    return None


def _choose_key_positions(length: int, count: int) -> list[int] | None:
    steps = [1, 2]
    random.shuffle(steps)
    for step in steps:
        span = (int(count) - 1) * int(step) + 1
        if span <= int(length):
            start = random.randint(0, int(length) - span)
            return [int(start + i * step) for i in range(int(count))]
    return None


def _marker_counts(frame_count: int, key_count: int) -> list[int]:
    counts = [random.randint(1, int(key_count)) for _ in range(int(frame_count))]
    if frame_count > 1 and len(set(counts)) == 1:
        replacement = random.choice([i for i in range(1, int(key_count) + 1) if i != counts[0]])
        counts[random.randrange(int(frame_count))] = int(replacement)
    random.shuffle(counts)
    return counts


def _draw_input_scene(
    rows: int,
    cols: int,
    bg: int,
    frame_color: int,
    key_colors: list[int],
    orientation: str,
    marker_counts: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((int(rows), int(cols)), int(bg), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    if orientation == "bottom":
        place_rows, place_cols = int(rows - 2), int(cols)
    else:
        place_rows, place_cols = int(rows), int(cols - 2)

    placed: list[tuple[tuple[int, int, int, int], int]] = []
    for marker_count in sorted(marker_counts, reverse=True):
        size = _sample_frame_size(marker_count)
        if size is None:
            return None
        outer_h, outer_w = size
        r_hi = int(place_rows - outer_h - 1)
        c_hi = int(place_cols - outer_w - 1)
        if r_hi < 1 or c_hi < 1:
            return None

        bbox = None
        for _ in range(150):
            r0 = random.randint(1, r_hi)
            c0 = random.randint(1, c_hi)
            candidate = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if _can_place(blocked, candidate):
                bbox = candidate
                break
        if bbox is None:
            return None
        _reserve(blocked, bbox)
        placed.append((bbox, int(marker_count)))

    marker_palette = [c for c in range(COLOR_COUNT) if c not in (int(bg), int(frame_color))]
    for bbox, marker_count in placed:
        _paint_frame(grid, bbox, int(frame_color))
        cells = _sample_spaced_interior_cells(bbox, marker_count)
        if cells is None:
            return None
        colors = random.sample(marker_palette, int(marker_count))
        for (r, c), color in zip(cells, colors):
            grid[int(r), int(c)] = int(color)

    if orientation == "bottom":
        cols_for_keys = _choose_key_positions(cols, len(key_colors))
        if cols_for_keys is None:
            return None
        for c, color in zip(cols_for_keys, key_colors):
            grid[int(rows) - 1, int(c)] = int(color)
    else:
        rows_for_keys = _choose_key_positions(rows, len(key_colors))
        if rows_for_keys is None:
            return None
        for r, color in zip(rows_for_keys, key_colors):
            grid[int(r), int(cols) - 1] = int(color)

    expected = grid.copy()
    for bbox, marker_count in placed:
        _paint_frame(expected, bbox, key_colors[int(marker_count) - 1])
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_SIZE_RANGE)
    cols = random.randint(*GRID_SIZE_RANGE)
    bg, frame_color = random.sample(range(COLOR_COUNT), 2)

    key_limit = min(KEY_COUNT_RANGE[1], COLOR_COUNT - 2, rows, cols)
    if key_limit < KEY_COUNT_RANGE[0]:
        return None
    key_count = random.randint(KEY_COUNT_RANGE[0], key_limit)
    color_pool = [c for c in range(COLOR_COUNT) if c not in (int(bg), int(frame_color))]
    key_colors = random.sample(color_pool, int(key_count))

    frame_count = random.randint(*FRAME_COUNT_RANGE)
    counts = _marker_counts(frame_count, key_count)
    orientation = random.choice(["bottom", "right"])
    pair = _draw_input_scene(rows, cols, bg, frame_color, key_colors, orientation, counts)
    if pair is None:
        return None
    grid, expected = pair

    if not _dominant_bg(grid, bg):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid key-indexed frame recoloring input")
