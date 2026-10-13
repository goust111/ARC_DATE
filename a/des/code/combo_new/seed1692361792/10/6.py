from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_STEP = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _most_frequent_non_bg(grid: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    assert colors.size > 0, "expected at least one foreground color"
    return int(colors[int(np.argmax(counts))])


def _is_consecutive(values: list[int]) -> bool:
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))


def _gap_candidate(
    grid: np.ndarray,
    border: int,
    line: list[tuple[int, int]],
    marker_cells: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int] | None:
    missing = [(r, c) for r, c in line if int(grid[r, c]) != border]
    if len(missing) not in (1, 2):
        return None
    anchors = [c if len({r for r, _c in line}) == 1 else r for r, c in missing]
    if not _is_consecutive(anchors):
        return None
    if any(int(grid[r, c]) != BG for r, c in missing):
        return None
    missing_set = set(missing)
    if any((r, c) not in missing_set and int(grid[r, c]) != border for r, c in line):
        return None

    colored = [
        int(grid[r, c])
        for r, c in marker_cells
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) not in (BG, border)
    ]
    if len(colored) != 1:
        return None
    return missing, int(colored[0])


def _frame_info(grid: np.ndarray, border: int, coords: np.ndarray) -> dict[str, object] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if any(int(grid[r, c]) != border for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
        return None

    candidates: list[tuple[str, list[tuple[int, int]], int]] = []

    top_line = [(r0, c) for c in range(c0 + 1, c1)]
    top_marker = [(r0 - 1, c) for _r, c in top_line]
    top = _gap_candidate(grid, border, top_line, top_marker)
    if top is not None:
        candidates.append(("top", top[0], top[1]))

    bottom_line = [(r1, c) for c in range(c0 + 1, c1)]
    bottom_marker = [(r1 + 1, c) for _r, c in bottom_line]
    bottom = _gap_candidate(grid, border, bottom_line, bottom_marker)
    if bottom is not None:
        candidates.append(("bottom", bottom[0], bottom[1]))

    left_line = [(r, c0) for r in range(r0 + 1, r1)]
    left_marker = [(r, c0 - 1) for r, _c in left_line]
    left = _gap_candidate(grid, border, left_line, left_marker)
    if left is not None:
        candidates.append(("left", left[0], left[1]))

    right_line = [(r, c1) for r in range(r0 + 1, r1)]
    right_marker = [(r, c1 + 1) for r, _c in right_line]
    right = _gap_candidate(grid, border, right_line, right_marker)
    if right is not None:
        candidates.append(("right", right[0], right[1]))

    if len(candidates) != 1:
        return None

    side, gap_cells, marker_color = candidates[0]
    expected_area = 2 * h + 2 * w - 4 - len(gap_cells)
    if int(coords.shape[0]) != expected_area:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_cells": gap_cells,
        "marker_color": marker_color,
    }


def _paint_rays(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    gap_cells: list[tuple[int, int]],
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        for _r, c in gap_cells:
            out[r0:r1, c] = color
        return
    if side == "bottom":
        for _r, c in gap_cells:
            out[r1 : r0 : -1, c] = color
        return
    if side == "left":
        for r, _c in gap_cells:
            out[r, c0:c1] = color
        return
    for r, _c in gap_cells:
        out[r, c1:c0:-1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each gapped rectangular frame and shoot a colored ray inward from its opening.

    Input structure:
    - The background is black.
    - One shared frame color draws several one-cell-thick hollow rectangular frames.
    - Every frame has exactly one gap of one or two consecutive cells on one side,
      and a single colored marker pixel sits immediately outside that opening.

    Rule:
    - Detect each frame, locate its unique gap, and determine which side of the
      rectangle contains the opening.
    - Read the color of the marker pixel adjacent to the gap on the outside.
    - Starting at the black gap cell or cells, paint a straight ray perpendicular
      to that side and extending through the interior until just before the
      opposite frame wall.
    - Keep the frame walls and the outside marker pixels unchanged.

    Key insight:
    - The missing border segment gives both the ray origin and its direction,
      while the adjacent outside marker supplies the ray color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    border = _most_frequent_non_bg(g, BG)
    components = _components(g == border)

    frames: list[dict[str, object]] = []
    for coords in components:
        info = _frame_info(g, border, coords)
        if info is not None:
            frames.append(info)

    assert frames, "expected at least one valid gapped frame"

    out = g.copy()
    for info in sorted(frames, key=lambda item: tuple(item["bbox"])):  # deterministic paint order
        _paint_rays(
            out,
            info["bbox"],
            str(info["side"]),
            list(info["gap_cells"]),
            int(info["marker_color"]),
        )
    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 30
FRAME_COUNT_RANGE = (1, 4)
FRAME_SIZE_RANGE = (4, 9)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _gap_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
    gap_len: int,
) -> list[tuple[int, int]]:
    if side == "top":
        start = random.randint(c0 + 1, c1 - gap_len)
        return [(r0, c) for c in range(start, start + gap_len)]
    if side == "bottom":
        start = random.randint(c0 + 1, c1 - gap_len)
        return [(r1, c) for c in range(start, start + gap_len)]
    if side == "left":
        start = random.randint(r0 + 1, r1 - gap_len)
        return [(r, c0) for r in range(start, start + gap_len)]
    start = random.randint(r0 + 1, r1 - gap_len)
    return [(r, c1) for r in range(start, start + gap_len)]


def _marker_cell(side: str, gap_cells: list[tuple[int, int]]) -> tuple[int, int]:
    r, c = random.choice(gap_cells)
    dr, dc = SIDE_TO_STEP[side]
    return r - dr, c - dc


def _draw_frame(
    grid: np.ndarray,
    border: int,
    bbox: tuple[int, int, int, int],
    side: str,
    gap_len: int,
    marker_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = border
    grid[r1, c0 : c1 + 1] = border
    grid[r0 : r1 + 1, c0] = border
    grid[r0 : r1 + 1, c1] = border

    gap = _gap_cells(r0, c0, r1, c1, side, gap_len)
    for r, c in gap:
        grid[r, c] = BG
    mr, mc = _marker_cell(side, gap)
    grid[mr, mc] = marker_color


def _can_reserve(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return not bool(used[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any())


def _reserve(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    used[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True


def _gen_attempt() -> np.ndarray | None:
    border = random.randint(1, 9)
    marker_pool = [color for color in range(1, 10) if color != border]
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    if frame_count > len(marker_pool):
        return None

    min_side = max(GRID_MIN, 7 + 4 * frame_count)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    used = np.zeros((h, w), dtype=bool)

    marker_colors = random.sample(marker_pool, frame_count)
    frame_specs: list[tuple[tuple[int, int, int, int], str, int, int]] = []

    for marker_color in marker_colors:
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            fh = random.randint(*FRAME_SIZE_RANGE)
            fw = random.randint(*FRAME_SIZE_RANGE)
            if fh > h - 2 or fw > w - 2:
                continue
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            if not _can_reserve(used, r0, c0, r1, c1):
                continue
            side = random.choice(SIDES)
            gap_len = random.choice((1, 2))
            _reserve(used, r0, c0, r1, c1)
            frame_specs.append(((r0, c0, r1, c1), side, gap_len, marker_color))
            placed = True
            break
        if not placed:
            return None

    random.shuffle(frame_specs)
    for bbox, side, gap_len, marker_color in frame_specs:
        _draw_frame(grid, border, bbox, side, gap_len, marker_color)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-frame ray puzzle")
