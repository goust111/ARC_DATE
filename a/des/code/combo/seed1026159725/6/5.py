from __future__ import annotations

import random

import numpy as np


RED = 2
SIDES = ("left", "right", "top", "bottom")
PALETTE = tuple(range(10))

GRID_HEIGHT_RANGE = (14, 28)
GRID_WIDTH_RANGE = (14, 28)
FRAME_COUNT_CHOICES = (1, 2, 3, 4)
FRAME_HEIGHT_RANGE = (4, 8)
FRAME_WIDTH_RANGE = (4, 8)
PLACEMENT_TRIES = 200
GENERATION_TRIES = 300
PAD = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        bbox = _rect_frame_bbox(grid == color)
        if bbox is not None:
            frames.append((color, bbox))
    assert frames, "expected at least one rectangular frame"
    return sorted(frames, key=lambda item: item[1])


def _segment_info(
    marker_coords: np.ndarray, frame_bbox: tuple[int, int, int, int]
) -> tuple[str, int, int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    rows = sorted({int(r) for r, _c in marker_coords})
    cols = sorted({int(c) for _r, c in marker_coords})

    if len(rows) == 1:
        row = rows[0]
        c0 = cols[0]
        c1 = cols[-1]
        length = c1 - c0 + 1
        assert len(cols) == length == marker_coords.shape[0], "horizontal segment must be contiguous"
        touches_left = c0 == fc0 + 1
        touches_right = c1 == fc1 - 1
        assert touches_left ^ touches_right, "horizontal segment must touch exactly one side wall"
        side = "left" if touches_left else "right"
        return side, row, c0, length

    if len(cols) == 1:
        col = cols[0]
        r0 = rows[0]
        r1 = rows[-1]
        length = r1 - r0 + 1
        assert len(rows) == length == marker_coords.shape[0], "vertical segment must be contiguous"
        touches_top = r0 == fr0 + 1
        touches_bottom = r1 == fr1 - 1
        assert touches_top ^ touches_bottom, "vertical segment must touch exactly one side wall"
        side = "top" if touches_top else "bottom"
        return side, col, r0, length

    raise AssertionError("marker must be a straight horizontal or vertical segment")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each internal red segment through its frame wall and draw an equally long external segment in the frame color.

    Input structure:
    - The background is the dominant color.
    - The grid contains one-pixel-thick rectangular frames of distinct colors.
    - Each frame contains one red straight segment whose touching endpoint reaches one inner wall.

    Rule:
    - Detect the red segment inside each frame.
    - Determine which wall it touches.
    - Draw a same-length segment immediately outside that wall, extending away from the frame, using the frame color.
    - Erase the internal red segment to background.
    - Keep every frame unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    frame_colors = {color for color, _bbox in frames}

    non_bg_colors = {int(color) for color in np.unique(g) if int(color) != bg}
    marker_colors = sorted(non_bg_colors - frame_colors)
    assert len(marker_colors) == 1, "expected a unique marker color distinct from frame colors"
    marker = marker_colors[0]

    out = g.copy()
    out[out == marker] = bg

    for frame_color, bbox in frames:
        fr0, fc0, fr1, fc1 = bbox
        interior = np.argwhere(g[fr0 + 1 : fr1, fc0 + 1 : fc1] == marker)
        assert interior.size > 0, "each frame must contain a marker segment"
        marker_coords = interior + np.array([fr0 + 1, fc0 + 1], dtype=int)

        side, fixed, start, length = _segment_info(marker_coords, bbox)

        if side == "left":
            row = fixed
            assert fc0 - length >= 0
            out[row, fc0 - length : fc0] = frame_color
        elif side == "right":
            row = fixed
            assert fc1 + length < g.shape[1]
            out[row, fc1 + 1 : fc1 + 1 + length] = frame_color
        elif side == "top":
            col = fixed
            assert fr0 - length >= 0
            out[fr0 - length : fr0, col] = frame_color
        else:
            col = fixed
            assert fr1 + length < g.shape[0]
            out[fr1 + 1 : fr1 + 1 + length, col] = frame_color

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _sample_sides(n_frames: int) -> list[str]:
    sides = list(random.sample(SIDES, k=n_frames))
    random.shuffle(sides)
    return sides


def _sample_spec(frame_color: int, side: str) -> dict[str, int | str]:
    if side in {"left", "right"}:
        height = random.randint(*FRAME_HEIGHT_RANGE)
        width = random.randint(max(5, FRAME_WIDTH_RANGE[0]), FRAME_WIDTH_RANGE[1])
        length = random.randint(2, width - 3)
        offset = random.randint(1, height - 2)
    else:
        height = random.randint(max(5, FRAME_HEIGHT_RANGE[0]), FRAME_HEIGHT_RANGE[1])
        width = random.randint(*FRAME_WIDTH_RANGE)
        length = random.randint(2, height - 3)
        offset = random.randint(1, width - 2)
    return {
        "color": frame_color,
        "side": side,
        "height": height,
        "width": width,
        "length": length,
        "offset": offset,
    }


def _reserve_bbox(spec: dict[str, int | str], r0: int, c0: int) -> tuple[int, int, int, int]:
    height = int(spec["height"])
    width = int(spec["width"])
    length = int(spec["length"])
    side = str(spec["side"])
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if side == "left":
        return r0, c0 - length, r1, c1
    if side == "right":
        return r0, c0, r1, c1 + length
    if side == "top":
        return r0 - length, c0, r1, c1
    return r0, c0, r1 + length, c1


def _placement_ranges(
    spec: dict[str, int | str], rows: int, cols: int
) -> tuple[int, int, int, int] | None:
    height = int(spec["height"])
    width = int(spec["width"])
    length = int(spec["length"])
    side = str(spec["side"])

    r_min = 0
    c_min = 0
    r_max = rows - height
    c_max = cols - width
    if side == "left":
        c_min = length
    elif side == "right":
        c_max = cols - width - length
    elif side == "top":
        r_min = length
    else:
        r_max = rows - height - length

    if r_min > r_max or c_min > c_max:
        return None
    return r_min, r_max, c_min, c_max


def _can_place(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    rows, cols = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rows - 1, r1 + pad)
    cc1 = min(cols - 1, c1 + pad)
    return not bool(np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _mark_occupied(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    rows, cols = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rows - 1, r1 + pad)
    cc1 = min(cols - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_marker(grid: np.ndarray, spec: dict[str, int | str]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    height = int(spec["height"])
    width = int(spec["width"])
    length = int(spec["length"])
    offset = int(spec["offset"])
    side = str(spec["side"])
    r1 = r0 + height - 1
    c1 = c0 + width - 1

    if side == "left":
        row = r0 + offset
        grid[row, c0 + 1 : c0 + 1 + length] = RED
    elif side == "right":
        row = r0 + offset
        grid[row, c1 - length : c1] = RED
    elif side == "top":
        col = c0 + offset
        grid[r0 + 1 : r0 + 1 + length, col] = RED
    else:
        col = c0 + offset
        grid[r1 - length : r1, col] = RED


def _draw_projection(grid: np.ndarray, spec: dict[str, int | str]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    height = int(spec["height"])
    width = int(spec["width"])
    length = int(spec["length"])
    offset = int(spec["offset"])
    color = int(spec["color"])
    side = str(spec["side"])
    r1 = r0 + height - 1
    c1 = c0 + width - 1

    if side == "left":
        row = r0 + offset
        grid[row, c0 - length : c0] = color
    elif side == "right":
        row = r0 + offset
        grid[row, c1 + 1 : c1 + 1 + length] = color
    elif side == "top":
        col = c0 + offset
        grid[r0 - length : r0, col] = color
    else:
        col = c0 + offset
        grid[r1 + 1 : r1 + 1 + length, col] = color


def _render_pair(rows: int, cols: int, bg: int, specs: list[dict[str, int | str]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), bg, dtype=int)
    out = np.full((rows, cols), bg, dtype=int)
    for spec in specs:
        _draw_frame(inp, int(spec["r0"]), int(spec["c0"]), int(spec["height"]), int(spec["width"]), int(spec["color"]))
        _draw_frame(out, int(spec["r0"]), int(spec["c0"]), int(spec["height"]), int(spec["width"]), int(spec["color"]))
        _draw_marker(inp, spec)
        _draw_projection(out, spec)
    return inp, out


def _place_specs(rows: int, cols: int, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    frame_colors = random.sample([c for c in PALETTE if c not in {bg, RED}], k=n_frames)
    specs = [_sample_spec(color, side) for color, side in zip(frame_colors, _sample_sides(n_frames))]

    occupied = np.zeros((rows, cols), dtype=bool)
    placed: list[dict[str, int | str]] = []
    for spec in specs:
        ranges = _placement_ranges(spec, rows, cols)
        if ranges is None:
            return None

        r_min, r_max, c_min, c_max = ranges
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            bbox = _reserve_bbox(spec, r0, c0)
            if not _can_place(occupied, bbox, PAD):
                continue
            _mark_occupied(occupied, bbox, PAD)
            placed_spec = dict(spec)
            placed_spec["r0"] = r0
            placed_spec["c0"] = c0
            placed.append(placed_spec)
            break
        else:
            return None

    inp, out = _render_pair(rows, cols, bg, placed)
    colors, counts = np.unique(inp, return_counts=True)
    bg_count = int(counts[np.where(colors == bg)[0][0]])
    if any(int(count) >= bg_count for color, count in zip(colors, counts) if int(color) != bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        rows = random.randint(*GRID_HEIGHT_RANGE)
        cols = random.randint(*GRID_WIDTH_RANGE)
        bg = random.choice([color for color in PALETTE if color != RED])
        pair = _place_specs(rows, cols, bg)
        if pair is not None:
            inp, _out = pair
            return inp
    raise RuntimeError("failed to generate a valid input")
