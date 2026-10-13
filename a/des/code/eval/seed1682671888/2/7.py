from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _split_regions(grid: np.ndarray, bg: int) -> tuple[int, list[np.ndarray]]:
    height, width = grid.shape
    candidates: dict[int, list[int]] = {}
    for c in range(width):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != int(bg):
            candidates.setdefault(int(col[0]), []).append(int(c))
    sep_color, sep_cols = max(candidates.items(), key=lambda item: len(item[1]))
    assert len(sep_cols) >= 1

    regions: list[np.ndarray] = []
    start = 0
    for c in sorted(sep_cols):
        if start < c:
            regions.append(grid[:, start:c])
        start = c + 1
    if start < width:
        regions.append(grid[:, start:width])
    assert len(regions) >= 2
    assert all(region.size > 0 for region in regions)
    return int(sep_color), regions


def _non_bg_bbox(region: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(region != int(bg))
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(region: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    for color in np.unique(region):
        color = int(color)
        if color == int(bg):
            continue
        coords = np.argwhere(region == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 4 or width < 4:
            continue
        expected = 2 * (height + width) - 4
        if int(coords.shape[0]) != expected:
            continue
        patch = region[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(patch[border] == color) and np.all(patch[~border] != color):
            return color, (int(r0), int(c0), int(r1), int(c1))
    raise AssertionError("frame not found")


def _arrow_direction(
    region: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    frame_color: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = frame_bbox
    mask = np.zeros_like(region, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    outside = (region != int(bg)) & (~mask) & (region != int(frame_color))
    coords = np.argwhere(outside)
    if coords.size == 0:
        return None

    colors = [int(region[r, c]) for r, c in coords]
    counts: dict[int, int] = {}
    for color in colors:
        counts[color] = counts.get(color, 0) + 1
    arrow_color = max(counts, key=counts.get)
    arrow = np.argwhere(outside & (region == int(arrow_color)))
    assert int(arrow.shape[0]) == 3

    rr0, cc0 = arrow.min(axis=0)
    rr1, cc1 = arrow.max(axis=0)
    if int(cc1) < int(c0):
        return (0, -1)
    if int(cc0) > int(c1):
        return (0, 1)
    if int(rr1) < int(r0):
        return (-1, 0)
    if int(rr0) > int(r1):
        return (1, 0)
    raise AssertionError("arrow must lie on one side of the frame")


def _shift(canvas: np.ndarray, direction: tuple[int, int], bg: int) -> np.ndarray:
    dr, dc = direction
    height, width = canvas.shape
    out = np.full((height, width), int(bg), dtype=int)

    src_r0 = max(0, -dr)
    src_r1 = min(height, height - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(width, width - dc)
    dst_r0 = src_r0 + dr
    dst_r1 = src_r1 + dr
    dst_c0 = src_c0 + dc
    dst_c1 = src_c1 + dc
    out[dst_r0:dst_r1, dst_c0:dst_c1] = canvas[src_r0:src_r1, src_c0:src_c1]
    return out


def _parse_instruction(
    region: np.ndarray,
    bg: int,
    canvas_shape: tuple[int, int],
) -> tuple[tuple[int, int] | None, list[tuple[int, int, int]]]:
    frame_color, frame_bbox = _find_frame(region, bg)
    r0, c0, r1, c1 = frame_bbox
    inner = region[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape == canvas_shape

    overlay: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(inner != int(bg)):
        overlay.append((int(r), int(c), int(inner[r, c])))

    direction = _arrow_direction(region, bg, frame_bbox, frame_color)
    return direction, overlay


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    _sep_color, regions = _split_regions(g, bg)

    init_region = regions[0]
    ir0, ic0, ir1, ic1 = _non_bg_bbox(init_region, bg)
    canvas = init_region[ir0 : ir1 + 1, ic0 : ic1 + 1].copy()

    for region in regions[1:]:
        direction, overlay = _parse_instruction(region, bg, canvas.shape)
        if direction is not None:
            canvas = _shift(canvas, direction, bg)
        for r, c, color in overlay:
            canvas[r, c] = int(color)
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read separator-delimited instruction regions from left to right.

    The first region is the initial canvas. Each later region contains a frame
    whose interior encodes overwrite pixels; if a small 3-pixel marker sits
    outside the frame, the current canvas is first shifted by one cell toward
    that marker's side, then the overwrite pixels are applied.
    """
    return _solve(grid)


_CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _random_canvas(height: int, width: int, colors: list[int]) -> np.ndarray:
    canvas = np.array([[random.choice(colors) for _ in range(width)] for _ in range(height)], dtype=int)
    if len(set(int(v) for v in canvas.ravel())) == 1:
        rr = random.randrange(height)
        cc = random.randrange(width)
        canvas[rr, cc] = random.choice([color for color in colors if color != int(canvas[rr, cc])])
    return canvas


def _overlay_cells(height: int, width: int, colors: list[int]) -> list[tuple[int, int, int]]:
    total = height * width
    count = random.randint(1, max(1, min(total, (total + 1) // 2)))
    cells = random.sample([(r, c) for r in range(height) for c in range(width)], count)
    return [(int(r), int(c), int(random.choice(colors))) for r, c in cells]


def _arrow_cells(
    frame_top: int,
    frame_left: int,
    frame_bottom: int,
    frame_right: int,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    mid_r = (frame_top + frame_bottom) // 2
    mid_c = (frame_left + frame_right) // 2
    dr, dc = direction
    if (dr, dc) == (0, -1):
        top = max(0, min(mid_r, frame_bottom - 1))
        left = frame_left - 3
        return [(top, left + 1), (top + 1, left), (top + 1, left + 1)]
    if (dr, dc) == (0, 1):
        top = max(0, min(mid_r, frame_bottom - 1))
        left = frame_right + 2
        return [(top, left), (top + 1, left), (top + 1, left + 1)]
    if (dr, dc) == (-1, 0):
        top = frame_top - 3
        left = max(0, min(mid_c, frame_right - 1))
        return [(top + 1, left), (top + 1, left + 1), (top, left + 1)]
    top = frame_bottom + 2
    left = max(0, min(mid_c, frame_right - 1))
    return [(top, left), (top, left + 1), (top + 1, left + 1)]


def _render_instruction(
    region_shape: tuple[int, int],
    bg: int,
    frame_color: int,
    arrow_color: int,
    canvas_shape: tuple[int, int],
    pad: int,
    overlay: list[tuple[int, int, int]],
    direction: tuple[int, int] | None,
) -> np.ndarray:
    region = np.full(region_shape, int(bg), dtype=int)
    canvas_h, canvas_w = canvas_shape
    frame_top = pad
    frame_left = pad
    frame_bottom = frame_top + canvas_h + 1
    frame_right = frame_left + canvas_w + 1

    region[frame_top, frame_left : frame_right + 1] = int(frame_color)
    region[frame_bottom, frame_left : frame_right + 1] = int(frame_color)
    region[frame_top : frame_bottom + 1, frame_left] = int(frame_color)
    region[frame_top : frame_bottom + 1, frame_right] = int(frame_color)

    for r, c, color in overlay:
        region[frame_top + 1 + r, frame_left + 1 + c] = int(color)

    if direction is not None:
        for r, c in _arrow_cells(frame_top, frame_left, frame_bottom, frame_right, direction):
            region[r, c] = int(arrow_color)
    return region


def _simulate(
    canvas: np.ndarray,
    ops: list[tuple[tuple[int, int] | None, list[tuple[int, int, int]]]],
    bg: int,
) -> np.ndarray:
    out = canvas.copy()
    for direction, overlay in ops:
        if direction is not None:
            out = _shift(out, direction, bg)
        for r, c, color in overlay:
            out[r, c] = int(color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        colors = list(range(10))
        bg = random.choice(colors)
        colors.remove(bg)
        sep_color, frame_color, arrow_color = random.sample(colors, 3)
        draw_colors = [color for color in colors if color not in {sep_color, frame_color, arrow_color}]
        random.shuffle(draw_colors)

        canvas_h = random.randint(2, 6)
        canvas_w = random.randint(2, 6)
        pad = random.randint(3, 4)
        canvas = _random_canvas(canvas_h, canvas_w, draw_colors[: max(3, min(len(draw_colors), 5))])

        n_instr = random.randint(2, 4)
        shift_flags = [random.choice([False, True]) for _ in range(n_instr)]
        if all(shift_flags):
            shift_flags[random.randrange(n_instr)] = False
        if not any(shift_flags):
            shift_flags[random.randrange(n_instr)] = True

        ops: list[tuple[tuple[int, int] | None, list[tuple[int, int, int]]]] = []
        used_dirs: list[tuple[int, int]] = []
        for use_shift in shift_flags:
            direction = random.choice(_CARDINALS) if use_shift else None
            if direction is not None:
                used_dirs.append(direction)
            overlay = _overlay_cells(canvas_h, canvas_w, draw_colors)
            ops.append((direction, overlay))

        frame_h = canvas_h + 2
        frame_w = canvas_w + 2
        region_h = frame_h + 2 * pad
        region_w = frame_w + 2 * pad
        total_w = (n_instr + 1) * region_w + n_instr
        grid = np.full((region_h, total_w), int(bg), dtype=int)

        initial_top = pad + 1
        initial_left = pad + 1
        grid[initial_top : initial_top + canvas_h, initial_left : initial_left + canvas_w] = canvas

        for idx in range(n_instr):
            sep_col = (idx + 1) * region_w + idx
            grid[:, sep_col] = int(sep_color)

        for idx, (direction, overlay) in enumerate(ops, start=1):
            start_col = idx * region_w + (idx - 1) + 1
            region = _render_instruction(
                (region_h, region_w),
                bg,
                frame_color,
                arrow_color,
                (canvas_h, canvas_w),
                pad,
                overlay,
                direction,
            )
            grid[:, start_col : start_col + region_w] = region

        expected = _simulate(canvas, ops, bg)
        if np.array_equal(expected, canvas):
            continue
        solved = _solve(grid)
        if solved.shape != (canvas_h, canvas_w):
            continue
        if not np.array_equal(solved, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid sample")
