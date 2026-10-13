from __future__ import annotations

import random

import numpy as np


OUTER_ORDER = ("nw", "ne", "se", "sw")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = grid.shape

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_rect_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if height < 4 or width < 4:
        return None
    if r0 < 1 or c0 < 1 or r1 + 1 >= grid.shape[0] or c1 + 1 >= grid.shape[1]:
        return None

    expected_area = 2 * (height + width) - 4
    if int(coords.shape[0]) != expected_area:
        return None

    coord_set = {tuple(map(int, rc)) for rc in coords.tolist()}
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == int(color)):
        return None
    return r0, c0, r1, c1


def _outer_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c1 + 1), (r1 + 1, c0 - 1)]


def _inner_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r0 + 1, c0 + 1), (r0 + 1, c1 - 1), (r1 - 1, c1 - 1), (r1 - 1, c0 + 1)]


def _rotate_clockwise(colors: list[int], shift: int) -> list[int]:
    return [int(colors[(idx - shift) % 4]) for idx in range(4)]


def _detect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    bg = _get_bg(grid)
    frame_bboxes: list[tuple[int, int, int, int]] = []
    frame_colors: set[int] = set()
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        for coords in _color_components(grid, color):
            bbox = _is_rect_frame(grid, color, coords)
            if bbox is not None:
                frame_bboxes.append(bbox)
                frame_colors.add(int(color))

    assert frame_bboxes, "expected at least one hollow frame"
    assert len(frame_colors) == 1, "all frames must share one border color"
    frame_bboxes.sort()
    return frame_bboxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each frame's four outside corner colors so the matching color lands at the inner dot's corner.

    Input structure:
    - The dominant background contains one or more 1-pixel-thick hollow rectangular frames of one shared color.
    - Each frame has four distinct colored pixels diagonally outside its four corners.
    - Inside each frame, exactly one colored dot sits on one of the four inner-corner cells.

    Rule:
    - For each frame, keep the frame and its inner dot fixed.
    - Treat the four outside corner colors as a clockwise cycle in NW, NE, SE, SW order.
    - Apply the unique cyclic rotation that moves the outside color matching the inner dot onto that same corner position.

    Key insight:
    - The inner dot gives a double address: its color chooses which outside marker must move, and its corner position chooses where the whole 4-color cycle must stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    out = g.copy()

    for bbox in _detect_frames(g):
        r0, c0, r1, c1 = bbox
        outer_positions = _outer_positions(bbox)
        inner_positions = _inner_positions(bbox)

        outer_colors = [int(g[r, c]) for r, c in outer_positions]
        assert all(color not in (bg, int(g[r0, c0])) for color in outer_colors)
        assert len(set(outer_colors)) == 4, "each frame needs four distinct outside colors"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colored_inside = np.argwhere(interior != bg)
        assert colored_inside.shape[0] == 1, "each frame interior must contain exactly one colored dot"
        local_r, local_c = map(int, colored_inside[0])
        dot_pos = (r0 + 1 + local_r, c0 + 1 + local_c)
        assert dot_pos in inner_positions, "the inner dot must sit on an inner corner"

        target_index = inner_positions.index(dot_pos)
        dot_color = int(g[dot_pos])
        assert dot_color in outer_colors, "the inner dot color must match one outside corner color"

        source_index = outer_colors.index(dot_color)
        shift = (target_index - source_index) % 4
        rotated = _rotate_clockwise(outer_colors, shift)
        for (r, c), color in zip(outer_positions, rotated):
            out[r, c] = int(color)

    return out


def _draw_frame_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_frame_bboxes(height: int, width: int, frame_count: int) -> list[tuple[int, int, int, int]] | None:
    bboxes: list[tuple[int, int, int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []

    for _ in range(frame_count):
        placed = False
        for _ in range(120):
            frame_h = random.randint(4, min(8, height - 2))
            if random.random() < 0.5:
                frame_w = frame_h
            else:
                frame_w = random.randint(4, min(8, width - 2))

            if frame_h >= height - 1 or frame_w >= width - 1:
                continue

            r0 = random.randint(1, height - frame_h - 1)
            c0 = random.randint(1, width - frame_w - 1)
            bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            reserve = (
                max(0, r0 - 2),
                max(0, c0 - 2),
                min(height - 1, bbox[2] + 2),
                min(width - 1, bbox[3] + 2),
            )
            if any(_rectangles_overlap(reserve, other) for other in reserved):
                continue

            reserved.append(reserve)
            bboxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    bboxes.sort()
    return bboxes


def _sample_shifts(frame_count: int) -> list[int]:
    shifts = [random.randrange(4) for _ in range(frame_count)]
    if all(shift == 0 for shift in shifts):
        shifts[random.randrange(frame_count)] = random.choice((1, 2, 3))
    elif frame_count > 1 and all(shift != 0 for shift in shifts):
        shifts[random.randrange(frame_count)] = 0
    return shifts


def _render_candidate(
    height: int,
    width: int,
    bg: int,
    frame_color: int,
    bboxes: list[tuple[int, int, int, int]],
    shifts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    palette = [color for color in range(10) if color not in (bg, frame_color)]

    for bbox, shift in zip(bboxes, shifts):
        _draw_frame_border(inp, bbox, frame_color)
        _draw_frame_border(out, bbox, frame_color)

        corner_colors = random.sample(palette, 4)
        target_index = random.randrange(4)
        dot_color = int(corner_colors[(target_index - shift) % 4])
        rotated = _rotate_clockwise(corner_colors, shift)

        for (r, c), color_in, color_out in zip(_outer_positions(bbox), corner_colors, rotated):
            inp[r, c] = int(color_in)
            out[r, c] = int(color_out)

        dot_r, dot_c = _inner_positions(bbox)[target_index]
        inp[dot_r, dot_c] = dot_color
        out[dot_r, dot_c] = dot_color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    max_frames = 1 + int(min(height, width) >= 20) + int(min(height, width) >= 24) + int(min(height, width) >= 28)
    frame_count = random.randint(1, max_frames)

    bboxes = _sample_frame_bboxes(height, width, frame_count)
    if bboxes is None:
        return None

    bg = random.randrange(10)
    frame_color = random.choice([color for color in range(10) if color != bg])
    shifts = _sample_shifts(frame_count)
    inp, expected = _render_candidate(height, width, bg, frame_color, bboxes, shifts)

    if _get_bg(inp) != bg:
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
