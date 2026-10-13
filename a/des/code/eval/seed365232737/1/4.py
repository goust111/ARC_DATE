from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 8, 9)
MAX_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _line_indices_from_counts(counts: np.ndarray) -> list[int]:
    unique = sorted({int(v) for v in counts})
    assert len(unique) >= 2, "expected at least two wall-count levels"

    best_gap = -1
    threshold = None
    for lo, hi in zip(unique, unique[1:]):
        gap = hi - lo
        if gap > best_gap:
            best_gap = gap
            threshold = (lo + hi) / 2.0

    assert threshold is not None and best_gap > 0, "failed to split line counts"
    indices = [i for i, v in enumerate(counts) if float(v) > threshold]
    assert len(indices) >= 2, "expected multiple grid lines"

    spacings = {b - a for a, b in zip(indices, indices[1:])}
    assert len(spacings) == 1, "grid lines must be equally spaced"
    return indices


def _extract_layout(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    colors, counts = np.unique(g[g != bg], return_counts=True)
    assert colors.size >= 1, "expected wall color and seeds"
    wall = int(colors[np.argmax(counts)])

    wall_mask = g == wall
    row_lines = _line_indices_from_counts(np.count_nonzero(wall_mask, axis=1))
    col_lines = _line_indices_from_counts(np.count_nonzero(wall_mask, axis=0))

    row_step = row_lines[1] - row_lines[0]
    col_step = col_lines[1] - col_lines[0]
    for idx, r in enumerate(row_lines):
        assert r == idx * row_step, "horizontal grid lines must be periodic from the top edge"
    for idx, c in enumerate(col_lines):
        assert c == idx * col_step, "vertical grid lines must be periodic from the left edge"

    room_h = row_step - 1
    room_w = col_step - 1
    assert room_h >= 1 and room_w >= 1

    rooms: list[dict[str, object]] = []
    for rr in range(len(row_lines) - 1):
        for cc in range(len(col_lines) - 1):
            r0 = row_lines[rr] + 1
            r1 = row_lines[rr + 1] - 1
            c0 = col_lines[cc] + 1
            c1 = col_lines[cc + 1] - 1
            assert r1 - r0 + 1 == room_h
            assert c1 - c0 + 1 == room_w

            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            seed_coords = np.argwhere((patch != bg) & (patch != wall))
            assert seed_coords.shape == (1, 2), "each room must contain exactly one seed"
            sr_local, sc_local = map(int, seed_coords[0])
            sr = r0 + sr_local
            sc = c0 + sc_local
            seed_color = int(g[sr, sc])

            top_wall = g[row_lines[rr], col_lines[cc] + 1 : col_lines[cc + 1]]
            bottom_wall = g[row_lines[rr + 1], col_lines[cc] + 1 : col_lines[cc + 1]]
            left_wall = g[row_lines[rr] + 1 : row_lines[rr + 1], col_lines[cc]]
            right_wall = g[row_lines[rr] + 1 : row_lines[rr + 1], col_lines[cc + 1]]

            sides = {
                "top": bool(np.any(top_wall != wall)),
                "bottom": bool(np.any(bottom_wall != wall)),
                "left": bool(np.any(left_wall != wall)),
                "right": bool(np.any(right_wall != wall)),
            }
            rooms.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "seed": (sr, sc),
                    "color": seed_color,
                    "sides": sides,
                }
            )

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "room_h": room_h,
        "room_w": room_w,
        "rooms": rooms,
    }


def _vertical_limits(
    row: int,
    top_open: bool,
    bottom_open: bool,
    r0: int,
    sr: int,
    r1: int,
    c0: int,
    sc: int,
    c1: int,
) -> tuple[int, int]:
    if row == sr:
        return c0, c1

    if row < sr and top_open:
        dist = sr - r0
        frac = 0.0 if dist == 0 else (row - r0) / dist
        left_extent = int(np.floor(frac * (sc - c0)))
        right_extent = int(np.floor(frac * (c1 - sc)))
        return sc - left_extent, sc + right_extent

    if row > sr and bottom_open:
        dist = r1 - sr
        frac = 0.0 if dist == 0 else (r1 - row) / dist
        left_extent = int(np.floor(frac * (sc - c0)))
        right_extent = int(np.floor(frac * (c1 - sc)))
        return sc - left_extent, sc + right_extent

    return c0, c1


def _horizontal_limits(
    col: int,
    left_open: bool,
    right_open: bool,
    r0: int,
    sr: int,
    r1: int,
    c0: int,
    sc: int,
    c1: int,
) -> tuple[int, int]:
    if col == sc:
        return r0, r1

    if col < sc and left_open:
        dist = sc - c0
        frac = 0.0 if dist == 0 else (col - c0) / dist
        top_extent = int(np.floor(frac * (sr - r0)))
        bottom_extent = int(np.floor(frac * (r1 - sr)))
        return sr - top_extent, sr + bottom_extent

    if col > sc and right_open:
        dist = c1 - sc
        frac = 0.0 if dist == 0 else (c1 - col) / dist
        top_extent = int(np.floor(frac * (sr - r0)))
        bottom_extent = int(np.floor(frac * (r1 - sr)))
        return sr - top_extent, sr + bottom_extent

    return r0, r1


def _room_fill_mask(bbox: tuple[int, int, int, int], seed: tuple[int, int], sides: dict[str, bool]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)

    for r in range(r0, r1 + 1):
        v_lo, v_hi = _vertical_limits(
            r,
            bool(sides["top"]),
            bool(sides["bottom"]),
            r0,
            sr,
            r1,
            c0,
            sc,
            c1,
        )
        for c in range(c0, c1 + 1):
            if c < v_lo or c > v_hi:
                continue
            h_lo, h_hi = _horizontal_limits(
                c,
                bool(sides["left"]),
                bool(sides["right"]),
                r0,
                sr,
                r1,
                c0,
                sc,
                c1,
            )
            if h_lo <= r <= h_hi:
                mask[r - r0, c - c0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each room's center seed into a shape controlled by wall gaps.

    The grid is partitioned by periodic one-pixel wall lines. Inside every room,
    each wall side is either closed (intact wall segment) or open (the segment
    contains at least one black gap). Closed sides leave that axis fully filled;
    open sides impose a linear taper from the room center to a single-pixel tip
    on that side. Intersecting the active tapers yields rectangles, diamonds,
    and mixed flat-edge / pointed shapes.
    """
    layout = _extract_layout(grid)
    g = np.asarray(layout["grid"], dtype=int)
    bg = int(layout["bg"])
    out = g.copy()

    for room in layout["rooms"]:
        bbox = room["bbox"]
        seed = room["seed"]
        color = int(room["color"])
        sides = room["sides"]
        r0, c0, r1, c1 = bbox
        mask = _room_fill_mask(bbox, seed, sides)

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[mask] = color

    return out


def _choose_grid_spec(rng: random.Random) -> tuple[int, int, int, int]:
    room_shapes = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
    room_rows, room_cols = rng.choice(room_shapes)

    possible_sizes: list[tuple[int, int]] = []
    for room_h in (5, 7, 9):
        for room_w in (5, 7, 9):
            total_h = room_rows * room_h + room_rows + 1
            total_w = room_cols * room_w + room_cols + 1
            if total_h <= 25 and total_w <= 25:
                possible_sizes.append((room_h, room_w))

    assert possible_sizes, "no valid room sizes"
    room_h, room_w = rng.choice(possible_sizes)
    return room_rows, room_cols, room_h, room_w


def _apply_gap(
    grid: np.ndarray,
    orientation: str,
    line_index: int,
    start: int,
    end: int,
    rng: random.Random,
) -> bool:
    segment_len = end - start - 1
    assert segment_len >= 1

    if segment_len == 1:
        gap_len = 1
    else:
        gap_len = rng.choice([1, min(3, segment_len)])
        if gap_len >= segment_len:
            gap_len = segment_len - 1
        gap_len = max(1, gap_len)

    gap_start_min = start + 1
    gap_start_max = end - gap_len
    if gap_start_min > gap_start_max:
        return False

    gap_start = rng.randint(gap_start_min, gap_start_max)
    gap_end = gap_start + gap_len
    if orientation == "h":
        grid[line_index, gap_start:gap_end] = BG
    else:
        grid[gap_start:gap_end, line_index] = BG
    return True


def _pattern_summary(layout: dict[str, object]) -> tuple[int, int]:
    rooms = layout["rooms"]
    open_sides = 0
    room_patterns = set()
    for room in rooms:
        sides = room["sides"]
        key = tuple(int(bool(sides[name])) for name in ("top", "bottom", "left", "right"))
        room_patterns.add(key)
        open_sides += sum(key)
    return open_sides, len(room_patterns)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows, room_cols, room_h, room_w = _choose_grid_spec(rng)
    total_h = room_rows * room_h + room_rows + 1
    total_w = room_cols * room_w + room_cols + 1

    grid = np.full((total_h, total_w), BG, dtype=int)
    row_lines = [idx * (room_h + 1) for idx in range(room_rows + 1)]
    col_lines = [idx * (room_w + 1) for idx in range(room_cols + 1)]

    grid[row_lines, :] = WALL
    grid[:, col_lines] = WALL

    all_segments: list[tuple[str, int, int, int]] = []
    for r in row_lines:
        for c0, c1 in zip(col_lines, col_lines[1:]):
            all_segments.append(("h", r, c0, c1))
    for c in col_lines:
        for r0, r1 in zip(row_lines, row_lines[1:]):
            all_segments.append(("v", c, r0, r1))

    rng.shuffle(all_segments)
    for orientation, line_index, start, end in all_segments:
        if rng.random() < 0.5:
            _apply_gap(grid, orientation, line_index, start, end, rng)

    room_count = room_rows * room_cols
    seed_colors = rng.sample(SEED_COLORS, room_count)
    color_idx = 0
    for rr in range(room_rows):
        for cc in range(room_cols):
            sr = row_lines[rr] + 1 + room_h // 2
            sc = col_lines[cc] + 1 + room_w // 2
            grid[sr, sc] = seed_colors[color_idx]
            color_idx += 1

    try:
        layout = _extract_layout(grid)
    except AssertionError:
        return None

    if len(layout["rooms"]) != room_count:
        return None

    open_sides, pattern_count = _pattern_summary(layout)
    total_sides = 4 * room_count
    if not (0 < open_sides < total_sides):
        return None
    if pattern_count < 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-growth input")
