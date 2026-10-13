from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
OBJECT_COLORS = (2, 3, 5, 6, 7, 8, 9)
ARC_LIMIT = 30
MAX_WINDOWS = 4
MAX_GENERATION_ATTEMPTS = 300
ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _separator_rows(grid: np.ndarray, color: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == int(color)))]


def _separator_cols(grid: np.ndarray, color: int) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == int(color)))]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(separators) + [int(length)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        assert right - left > 1, "room spans must be non-empty"
        spans.append((int(left + 1), int(right)))
    return spans


def _room_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = _separator_rows(grid, BLUE)
    sep_cols = _separator_cols(grid, BLUE)
    assert sep_rows or sep_cols, "expected at least one divider line"

    row_spans = _spans(grid.shape[0], sep_rows)
    col_spans = _spans(grid.shape[1], sep_cols)

    row_heights = {r1 - r0 for r0, r1 in row_spans}
    col_widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_heights) == 1 and len(col_widths) == 1, "rooms should share one interior size"
    return row_spans, col_spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder divider-separated room contents by each room's yellow-window count.

    Input structure:
    - The background is black.
    - One-pixel blue rows and/or columns span the whole grid and partition it into a regular room lattice.
    - Each room contains one monochrome object in a unique color plus zero to four yellow single-pixel windows.

    Rule:
    - Read the rooms in row-major order from top-left to bottom-right.
    - Count the yellow pixels inside each room.
    - Stable-sort the room contents by that window count and place the sorted contents back into the rooms in row-major order.
    - Copy each room interior without changing the relative positions of its object and windows; keep all blue divider lines unchanged.

    Key insight:
    - The blue grid is a fixed set of containers, while the movable payload is the full room interior; the hidden sorting key is the number of yellow window pixels, not object shape or color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans = _room_grid(g)

    room_bounds: list[tuple[int, int, int, int]] = []
    room_patches: list[np.ndarray] = []
    window_counts: list[int] = []

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0:r1, c0:c1].copy()
            assert patch.size > 0
            assert not bool(np.any(patch == BLUE)), "divider color should not appear inside rooms"
            room_bounds.append((int(r0), int(r1), int(c0), int(c1)))
            room_patches.append(patch)
            window_counts.append(int(np.sum(patch == YELLOW)))

    order = sorted(range(len(room_patches)), key=window_counts.__getitem__)
    out = g.copy()
    for dst_idx, src_idx in enumerate(order):
        r0, r1, c0, c1 = room_bounds[dst_idx]
        patch = room_patches[src_idx]
        assert patch.shape == (r1 - r0, c1 - c0)
        out[r0:r1, c0:c1] = patch
    return out


LAYOUT_MODES = ("single_row", "single_col", "matrix")
MATRIX_LAYOUTS = ((2, 2), (2, 3), (3, 2))


def _sample_layout() -> tuple[int, int]:
    mode = random.choice(LAYOUT_MODES)
    if mode == "single_row":
        return 1, random.randint(2, len(OBJECT_COLORS))
    if mode == "single_col":
        return random.randint(2, len(OBJECT_COLORS)), 1
    return random.choice(MATRIX_LAYOUTS)


def _sample_room_size(n_rows: int, n_cols: int) -> tuple[int, int] | None:
    max_h = min(8, (ARC_LIMIT - (n_rows - 1)) // n_rows)
    max_w = min(8, (ARC_LIMIT - (n_cols - 1)) // n_cols)
    choices = [
        (h, w)
        for h in range(3, max_h + 1)
        for w in range(3, max_w + 1)
        if h * w >= 12
    ]
    return random.choice(choices) if choices else None


def _sample_window_counts(n_rooms: int) -> list[int] | None:
    need_tie = random.random() < 0.5
    for _ in range(200):
        counts = [random.randint(0, MAX_WINDOWS) for _ in range(n_rooms)]
        if len(set(counts)) == 1:
            continue
        if counts == sorted(counts):
            continue
        if need_tie and len(set(counts)) == n_rooms:
            continue
        if not need_tie and n_rooms <= MAX_WINDOWS + 1 and len(set(counts)) < n_rooms:
            continue
        return counts
    return None


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _connected_mask(max_h: int, max_w: int) -> np.ndarray:
    for _ in range(100):
        h = random.randint(1, min(4, max_h))
        w = random.randint(1, min(4, max_w))
        if h == 1 and w == 1:
            continue

        max_area = min(h * w, 6)
        area = random.randint(2, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ORTHO_DIRS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((int(rr), int(cc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        cells = _normalize_cells(cells)
        out_h = max(r for r, _c in cells) + 1
        out_w = max(c for _r, c in cells) + 1
        mask = np.zeros((out_h, out_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to sample a connected mask")


def _build_room(room_h: int, room_w: int, color: int, window_count: int) -> np.ndarray | None:
    room = np.full((room_h, room_w), BLACK, dtype=int)
    mask = _connected_mask(room_h, room_w)
    mh, mw = mask.shape
    top = random.randint(0, room_h - mh)
    left = random.randint(0, room_w - mw)
    room_slice = room[top : top + mh, left : left + mw]
    room_slice[mask] = int(color)

    free = [(int(r), int(c)) for r, c in zip(*np.where(room == BLACK), strict=False)]
    if len(free) < window_count:
        return None
    for r, c in random.sample(free, window_count):
        room[r, c] = YELLOW
    return room


def _assemble_grid(
    n_rows: int,
    n_cols: int,
    room_h: int,
    room_w: int,
    colors: list[int],
    window_counts: list[int],
) -> np.ndarray | None:
    height = n_rows * room_h + (n_rows - 1)
    width = n_cols * room_w + (n_cols - 1)
    if height > ARC_LIMIT or width > ARC_LIMIT:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for r in range(1, n_rows):
        grid[r * room_h + (r - 1), :] = BLUE
    for c in range(1, n_cols):
        grid[:, c * room_w + (c - 1)] = BLUE

    idx = 0
    for rr in range(n_rows):
        for cc in range(n_cols):
            room = _build_room(room_h, room_w, colors[idx], window_counts[idx])
            if room is None:
                return None
            r0 = rr * (room_h + 1)
            c0 = cc * (room_w + 1)
            grid[r0 : r0 + room_h, c0 : c0 + room_w] = room
            idx += 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = _sample_layout()
    n_rooms = n_rows * n_cols

    room_size = _sample_room_size(n_rows, n_cols)
    if room_size is None:
        return None
    room_h, room_w = room_size

    window_counts = _sample_window_counts(n_rooms)
    if window_counts is None:
        return None

    colors = random.sample(list(OBJECT_COLORS), n_rooms)
    grid = _assemble_grid(n_rows, n_cols, room_h, room_w, colors, window_counts)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if grid.shape != out.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-sorting puzzle")
