import random
from collections import Counter

import numpy as np


BG = 0
ANCHOR_COLOR = 9  # maroon
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _draw_outline(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _draw_filled(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top : top + height, left : left + width] = color


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == ANCHOR_COLOR)
    assert pts.shape[0] == 1, "expected a unique maroon anchor"
    return tuple(map(int, pts[0]))


def _find_band(grid: np.ndarray, bg: int, anchor: tuple[int, int]) -> tuple[int, list[int]]:
    h, w = grid.shape
    ar, ac = anchor
    for r in range(h - 1, -1, -1):
        cols = [c for c in range(w) if int(grid[r, c]) != bg and not (r == ar and c == ac)]
        if not 3 <= len(cols) <= 5:
            continue
        if any(cols[i] + 1 != cols[i + 1] for i in range(len(cols) - 1)):
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(colors) != len(set(colors)):
            continue
        return int(r), colors
    raise AssertionError("failed to locate the bottom color band")


def _extract_frames(grid: np.ndarray, band_row: int, anchor: tuple[int, int], bg: int) -> list[dict]:
    body = np.array(grid, copy=True)
    body[band_row, :] = bg
    body[anchor] = bg

    frames = []
    for color in sorted(set(map(int, np.unique(body))) - {bg}):
        pts = np.argwhere(body == color)
        if pts.size == 0:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 3 and width >= 3, "frame bbox must be at least 3x3"

        mask = body[r0 : r1 + 1, c0 : c1 + 1] == color
        expected = np.zeros((height, width), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(mask, expected), "each color must form a 1-pixel rectangular frame"

        frames.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
                "center_key": (r0 + r1, c0 + c1),
            }
        )

    frames.sort(key=lambda item: (-item["height"] * item["width"], -item["height"], -item["width"]))
    assert 3 <= len(frames) <= 5, "expected 3-5 nested frames"

    center_key = frames[0]["center_key"]
    for i, frame in enumerate(frames):
        assert frame["center_key"] == center_key, "all frames must share a center"
        if i == 0:
            continue
        prev = frames[i - 1]["bbox"]
        cur = frame["bbox"]
        assert cur[0] == prev[0] + 2 and cur[1] == prev[1] + 2
        assert cur[2] == prev[2] - 2 and cur[3] == prev[3] - 2

    return frames


def _paint_transformed(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    frames: list[dict],
    band_colors: list[int],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    ar, ac = anchor
    depth_by_color = {frame["color"]: depth for depth, frame in enumerate(frames)}

    for color in band_colors:
        depth = depth_by_color[color]
        frame = frames[depth]
        new_height = int(frame["height"] - 2 * depth)
        new_width = int(frame["width"] - 2 * depth)
        assert new_height >= 1 and new_width >= 1, "shrunk geometry must stay positive"
        assert ar + new_height <= shape[0] and ac + new_width <= shape[1], "transformed frame exceeds canvas"
        if new_height < 3 or new_width < 3:
            _draw_filled(out, ar, ac, new_height, new_width, color)
        else:
            _draw_outline(out, ar, ac, new_height, new_width, color)

    out[ar, ac] = ANCHOR_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose nested rectangular frames, shrink them by depth, and redraw them at the maroon anchor in band-defined stacking order.

    Input structure:
    - The background is black.
    - A single maroon anchor pixel sits in the upper-left area.
    - The main scene contains 3-5 concentric 1-pixel rectangular frames with one black gap between adjacent layers; each layer has a different color.
    - The bottom area contains one horizontal band of distinct colored pixels whose color set matches the frame colors but whose order is shuffled.

    Rule:
    - Detect the nested frames and sort them from outermost to innermost by their shared-center bounding boxes.
    - Read the bottom color band from left to right; that order is the draw order from lowest priority to highest priority.
    - For depth i (outermost depth 0), shrink that frame to height = original_height - 2*i and width = original_width - 2*i.
    - Redraw every shrunk shape with its top-left corner aligned to the maroon anchor; use a hollow frame when both shrunk dimensions are at least 3, otherwise use a filled rectangle of the shrunk size.
    - Clear the original frame cluster and the color band to black, then keep the maroon anchor visible on top.

    Key insight:
    - The nested geometry determines which layer gets which shrink amount, while the bottom color band does not remap colors to layers; it only controls the overpainting priority of the already-identified layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    anchor = _find_anchor(g)
    band_row, band_colors = _find_band(g, bg, anchor)
    frames = _extract_frames(g, band_row, anchor, bg)
    frame_colors = [int(frame["color"]) for frame in frames]
    assert Counter(frame_colors) == Counter(band_colors), "band colors must match the frame colors"
    return _paint_transformed(g.shape, anchor, frames, band_colors, bg)


def _random_band_order(rng: random.Random, colors: list[int]) -> list[int]:
    order = list(colors)
    while True:
        rng.shuffle(order)
        if order != colors:
            return list(order)


def _choose_outer_dim(
    rng: random.Random,
    layer_count: int,
    force_solid: bool,
    axis_limit: int,
    trigger_axis: bool,
) -> int:
    min_positive = 6 * (layer_count - 1) + 1
    min_frame = 6 * (layer_count - 1) + 3
    if force_solid and trigger_axis:
        high = min(axis_limit, min_positive + 1)
        return rng.randint(min_positive, high)
    return rng.randint(min_frame, axis_limit)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.randint(3, 5)
    force_solid = rng.random() < 0.5
    solid_mode = rng.choice(["height", "width", "both"]) if force_solid else "none"

    max_height = 27
    max_width = 28
    min_height_needed = 6 * (layer_count - 1) + 1
    min_width_needed = 6 * (layer_count - 1) + 1
    if min_height_needed > max_height or min_width_needed > max_width:
        return None

    outer_height = _choose_outer_dim(rng, layer_count, force_solid, max_height, solid_mode in {"height", "both"})
    outer_width = _choose_outer_dim(rng, layer_count, force_solid, max_width, solid_mode in {"width", "both"})

    extra_rows_max = 29 - outer_height
    extra_cols_max = 30 - outer_width
    if extra_rows_max < 2 or extra_cols_max < 2:
        return None

    canvas_height = outer_height + rng.randint(2, extra_rows_max) + 1
    canvas_width = outer_width + rng.randint(2, extra_cols_max)

    colors = rng.sample(FRAME_COLORS, layer_count)
    band_order = _random_band_order(rng, colors)
    grid = np.full((canvas_height, canvas_width), BG, dtype=int)

    frame_top = (canvas_height - 1 - outer_height) // 2
    frame_left = (canvas_width - outer_width) // 2
    anchor = (
        rng.randint(0, min(2, frame_top - 1)),
        rng.randint(0, min(2, frame_left - 1)),
    )
    grid[anchor] = ANCHOR_COLOR

    frames = []
    for depth, color in enumerate(colors):
        top = frame_top + 2 * depth
        left = frame_left + 2 * depth
        height = outer_height - 4 * depth
        width = outer_width - 4 * depth
        _draw_outline(grid, top, left, height, width, color)
        frames.append({"color": color, "height": height, "width": width})

    band_start = rng.randint(0, canvas_width - layer_count)
    for offset, color in enumerate(band_order):
        grid[canvas_height - 1, band_start + offset] = color

    expected = _paint_transformed(grid.shape, anchor, frames, band_order, BG)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
