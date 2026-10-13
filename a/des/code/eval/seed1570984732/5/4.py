import random
from collections import Counter

import numpy as np


SIDES = ("top", "bottom", "left", "right")
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SOURCE_COLOR = 9
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _side_center(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    cr, cc = _center_of_bbox(bbox)
    if side == "top":
        return r0, cc
    if side == "bottom":
        return r1, cc
    if side == "left":
        return cr, c0
    assert side == "right"
    return cr, c1


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _parse_frame(grid: np.ndarray, color: int, bg: int) -> dict:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        raise ValueError("empty color")

    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if min(height, width) < 3 or height % 2 == 0 or width % 2 == 0:
        raise ValueError("frame dimensions must be odd and at least 3")

    gap_sides = [side for side in SIDES if int(grid[_side_center(bbox, side)]) == bg]
    if len(gap_sides) != 1:
        raise ValueError("expected exactly one centered gap")
    gap_side = gap_sides[0]
    gap_pos = _side_center(bbox, gap_side)

    expected_count = 2 * (height + width) - 4 - 1
    if int(coords.shape[0]) != expected_count:
        raise ValueError("unexpected frame pixel count")

    coord_set = {tuple(map(int, xy)) for xy in coords}
    for pos in _perimeter_coords(bbox):
        expected = bg if pos == gap_pos else color
        if int(grid[pos]) != expected:
            raise ValueError("perimeter mismatch")
        if pos != gap_pos and pos not in coord_set:
            raise ValueError("missing frame pixel")

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) == color:
                raise ValueError("frame color leaked into interior")

    return {
        "color": int(color),
        "bbox": bbox,
        "center": _center_of_bbox(bbox),
        "gap_side": gap_side,
        "gap_pos": gap_pos,
    }


def _extract_structure(grid: np.ndarray) -> tuple[int, tuple[int, int], int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected 2D grid")

    bg = _get_bg(g)
    unique_colors = [int(v) for v in np.unique(g) if int(v) != bg]
    frames: list[dict] = []
    for color in unique_colors:
        if int(np.count_nonzero(g == color)) <= 1:
            continue
        frames.append(_parse_frame(g, color, bg))

    if not (3 <= len(frames) <= 4):
        raise ValueError("expected 3 or 4 frames")

    frames.sort(
        key=lambda frame: (
            frame["bbox"][2] - frame["bbox"][0],
            frame["bbox"][3] - frame["bbox"][1],
        )
    )

    centers = {frame["center"] for frame in frames}
    if len(centers) != 1:
        raise ValueError("frames must be concentric")
    source_pos = frames[0]["center"]
    source_color = int(g[source_pos])
    if source_color == bg or source_color in {frame["color"] for frame in frames}:
        raise ValueError("center signal source missing")
    if int(np.count_nonzero(g == source_color)) != 1:
        raise ValueError("signal source must be unique")

    for inner, outer in zip(frames, frames[1:]):
        ir0, ic0, ir1, ic1 = inner["bbox"]
        or0, oc0, or1, oc1 = outer["bbox"]
        if not (or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1):
            raise ValueError("frames must be strictly nested")
        if min(ir0 - or0, ic0 - oc0, or1 - ir1, oc1 - ic1) < 3:
            raise ValueError("adjacent frames need at least two bg cells of spacing")

    return bg, source_pos, source_color, frames


def _draw_ray(
    out: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    stop_before: tuple[int, int] | None,
    color: int,
    protected: set[tuple[int, int]],
) -> None:
    height, width = out.shape
    dr, dc = step
    r = start[0] + dr
    c = start[1] + dc
    while 0 <= r < height and 0 <= c < width:
        if stop_before is not None and (r, c) == stop_before:
            break
        if (r, c) not in protected:
            out[r, c] = int(color)
        r += dr
        c += dc


def _paint_side(out: np.ndarray, frame: dict, side: str, color: int, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    if side == "top":
        out[r0, c0 : c1 + 1] = int(color)
    elif side == "bottom":
        out[r1, c0 : c1 + 1] = int(color)
    elif side == "left":
        out[r0 : r1 + 1, c0] = int(color)
    else:
        assert side == "right"
        out[r0 : r1 + 1, c1] = int(color)
    gap_r, gap_c = frame["gap_pos"]
    out[gap_r, gap_c] = int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a signal through nested rectangular frames.

    The source sits at the center of the innermost frame. Each frame has one
    centered gap. The current frame's gap direction determines where the signal
    ray exits next; when the ray reaches the next outer frame, that entire side
    is recolored to the current frame's original color. Then the next frame
    emits a new outward ray from its own gap using its own original color.
    """
    g = np.asarray(grid, dtype=int)
    bg, source_pos, source_color, frames = _extract_structure(g)

    out = g.copy()
    protected = {source_pos} | {frame["gap_pos"] for frame in frames}

    start = source_pos
    for idx, frame in enumerate(frames):
        side = frame["gap_side"]
        step = SIDE_TO_STEP[side]
        next_frame = frames[idx + 1] if idx + 1 < len(frames) else None
        stop_before = _side_center(next_frame["bbox"], side) if next_frame is not None else None
        _draw_ray(out, start, step, stop_before, int(frame["color"]), protected)

        if next_frame is not None:
            _paint_side(out, next_frame, side, int(frame["color"]), bg)
            start = next_frame["gap_pos"]

    out[source_pos] = int(source_color)
    for gap_r, gap_c in protected - {source_pos}:
        out[gap_r, gap_c] = int(bg)
    return out


def _draw_frame(
    grid: np.ndarray,
    center: tuple[int, int],
    half_h: int,
    half_w: int,
    color: int,
    gap_side: str,
    bg: int,
) -> dict:
    cr, cc = center
    bbox = (cr - half_h, cc - half_w, cr + half_h, cc + half_w)
    for r, c in _perimeter_coords(bbox):
        grid[r, c] = int(color)
    gap_pos = _side_center(bbox, gap_side)
    grid[gap_pos] = int(bg)
    return {"bbox": bbox, "gap_pos": gap_pos}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice([3, 4])
    if frame_count == 3:
        inner_half_h = rng.randint(2, 4)
        inner_half_w = rng.randint(2, 5)
        step_low, step_high = 3, 5
        pad_high = 4
    else:
        inner_half_h = rng.randint(2, 3)
        inner_half_w = rng.randint(2, 4)
        step_low, step_high = 3, 4
        pad_high = 3

    half_heights = [inner_half_h]
    half_widths = [inner_half_w]
    for _ in range(1, frame_count):
        half_heights.append(half_heights[-1] + rng.randint(step_low, step_high))
        half_widths.append(half_widths[-1] + rng.randint(step_low, step_high))

    outer_half_h = half_heights[-1]
    outer_half_w = half_widths[-1]

    top_pad = rng.randint(1, pad_high)
    bottom_pad = top_pad + rng.choice([-1, 0, 1])
    left_pad = rng.randint(1, pad_high)
    right_pad = left_pad + rng.choice([-1, 0, 1])
    if min(bottom_pad, right_pad) < 1:
        return None

    height = top_pad + bottom_pad + 2 * outer_half_h + 1
    width = left_pad + right_pad + 2 * outer_half_w + 1
    if not (13 <= height <= 37 and 13 <= width <= 37):
        return None

    bg = rng.choice([0, 1, 2, 5, 6])
    palette = [c for c in range(10) if c not in {bg, SOURCE_COLOR}]
    rng.shuffle(palette)
    frame_colors = palette[:frame_count]
    gap_sides = [rng.choice(SIDES) for _ in range(frame_count)]

    grid = np.full((height, width), bg, dtype=int)
    center = (top_pad + outer_half_h, left_pad + outer_half_w)

    for half_h, half_w, color, side in zip(half_heights, half_widths, frame_colors, gap_sides):
        _draw_frame(grid, center, half_h, half_w, color, side, bg)

    grid[center] = SOURCE_COLOR

    bg_count = int(np.count_nonzero(grid == bg))
    color_counts = Counter(map(int, grid.ravel()))
    if any(count >= bg_count for color, count in color_counts.items() if color != bg):
        return None

    try:
        parsed_bg, parsed_source, parsed_source_color, parsed_frames = _extract_structure(grid)
    except ValueError:
        return None

    if parsed_bg != bg or parsed_source != center or parsed_source_color != SOURCE_COLOR:
        return None
    if [frame["gap_side"] for frame in parsed_frames] != gap_sides:
        return None
    if {frame["color"] for frame in parsed_frames} != set(frame_colors):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
