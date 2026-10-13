from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
GREY = Color.GREY
_RNG = np.random.default_rng()
_PALETTE_ANCHORS = ((0, 0), (0, 1))
_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # left, down, right, up


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _palette_colors(grid: np.ndarray, bg: int) -> tuple[int, int]:
    horizontal = int(grid[0, 0])
    vertical = int(grid[0, 1])
    assert horizontal != bg and vertical != bg
    assert horizontal != vertical
    return horizontal, vertical


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_hollow_frame(component: dict, palette_mask: np.ndarray) -> bool:
    coords = component["coords"]
    if np.any(palette_mask[coords[:, 0], coords[:, 1]]):
        return False

    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    border = _border_mask((h, w))
    if int(component["area"]) != int(border.sum()):
        return False
    return bool(np.array_equal(_component_mask(component), border))


def _trace_spiral(start: tuple[int, int], lengths: list[int]) -> list[tuple[int, int, int]]:
    r, c = map(int, start)
    visited = {(r, c)}
    trace: list[tuple[int, int, int]] = []
    for seg_idx, length in enumerate(lengths):
        dr, dc = _DIRS[seg_idx % 4]
        for _ in range(int(length)):
            r += dr
            c += dc
            assert (r, c) not in visited
            visited.add((r, c))
            trace.append((r, c, seg_idx))
    return trace


def _render_output(
    shape: tuple[int, int],
    bg: int,
    horizontal_color: int,
    vertical_color: int,
    start_color: int,
    start: tuple[int, int],
    lengths: list[int],
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    out[0, 0] = int(horizontal_color)
    out[0, 1] = int(vertical_color)
    sr, sc = map(int, start)
    out[sr, sc] = int(start_color)

    for r, c, seg_idx in _trace_spiral(start, lengths):
        assert 0 <= r < h and 0 <= c < w
        dr, _dc = _DIRS[seg_idx % 4]
        color = horizontal_color if dr == 0 else vertical_color
        assert (r, c) != (sr, sc)
        out[r, c] = int(color)
    return out


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    horizontal_color, vertical_color = _palette_colors(g, bg)

    palette_mask = np.zeros_like(g, dtype=bool)
    for r, c in _PALETTE_ANCHORS:
        palette_mask[r, c] = True

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    frame_border_mask = np.zeros_like(g, dtype=bool)
    for component in components:
        if _is_hollow_frame(component, palette_mask):
            coords = component["coords"]
            frame_border_mask[coords[:, 0], coords[:, 1]] = True
            frames.append(component)

    assert len(frames) >= 2

    interior_marker_mask = np.zeros_like(g, dtype=bool)
    counts: list[int] = []
    for component in frames:
        r0, c0, r1, c1 = map(int, component["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = interior != bg
        count = int(marker_mask.sum())
        assert count > 0
        interior_marker_mask[r0 + 1 : r1, c0 + 1 : c1] |= marker_mask
        counts.append(count)

    assert len(counts) == len(set(counts))
    lengths = sorted(counts)

    residual = (g != bg) & ~palette_mask & ~frame_border_mask & ~interior_marker_mask
    start_coords = np.argwhere(residual)
    assert start_coords.shape == (1, 2)
    start = tuple(map(int, start_coords[0]))
    sr, sc = start
    r0 = max(0, sr - 1)
    r1 = min(g.shape[0], sr + 2)
    c0 = max(0, sc - 1)
    c1 = min(g.shape[1], sc + 2)
    assert int(np.count_nonzero(g[r0:r1, c0:c1] != bg)) == 1

    return {
        "bg": bg,
        "horizontal_color": horizontal_color,
        "vertical_color": vertical_color,
        "start_color": int(g[start]),
        "start": start,
        "lengths": lengths,
        "frames": frames,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the marker dots inside hollow frames, sort the counts, and draw a color-split outward spiral from the lone start pixel.

    Input structure:
    - The background is the dominant color.
    - The upper-left corner contains two adjacent palette pixels; the left one gives the horizontal spiral color and the right one gives the vertical spiral color.
    - Several monochrome hollow rectangular frames are scattered in the grid, and each frame encloses a distinct positive number of interior marker pixels.
    - One extra isolated pixel lies outside every frame and serves as the spiral start marker.

    Rule:
    - For each hollow frame, count all non-background pixels strictly inside the frame.
    - Sort those counts in ascending order to obtain the spiral segment lengths.
    - Starting from the isolated marker, draw segments of those lengths while turning left, down, right, up cyclically.
    - Paint horizontal segments with the left palette color and vertical segments with the right palette color; keep the original start-marker color unchanged.
    - Remove every frame and every interior marker from the output while preserving the two palette pixels.

    Key insight:
    - The rectangles are only a parameter table: their interior marker counts, once sorted, determine the step lengths of the final spiral.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _render_output(
        grid.shape,
        scene["bg"],
        scene["horizontal_color"],
        scene["vertical_color"],
        scene["start_color"],
        scene["start"],
        scene["lengths"],
    )


def _relative_spiral_bbox(lengths: list[int]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    cells = [(0, 0)]
    for r, c, _seg_idx in _trace_spiral((0, 0), lengths):
        cells.append((r, c))
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return cells, (min(rows), min(cols), max(rows), max(cols))


def _marker_slots(inner_h: int, inner_w: int) -> list[list[tuple[int, int]]]:
    slots: list[list[tuple[int, int]]] = []
    for pr in (0, 1):
        for pc in (0, 1):
            group = [(r, c) for r in range(inner_h) for c in range(inner_w) if r % 2 == pr and c % 2 == pc]
            if group:
                slots.append(group)
    return slots


def _feasible_frame_dims(count: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for inner_h in range(1, 7):
        for inner_w in range(1, 7):
            groups = _marker_slots(inner_h, inner_w)
            if max(len(group) for group in groups) >= count:
                dims.append((inner_h + 2, inner_w + 2))
    return dims


def _sample_marker_positions(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]]:
    groups = [group for group in _marker_slots(inner_h, inner_w) if len(group) >= count]
    assert groups
    group = groups[int(_RNG.integers(0, len(groups)))]
    pick = _RNG.choice(len(group), size=count, replace=False)
    return [group[int(idx)] for idx in pick]


def _place_box(
    blocked: np.ndarray,
    box_h: int,
    box_w: int,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates = [
        (r0, c0)
        for r0 in range(0, h - box_h + 1)
        for c0 in range(0, w - box_w + 1)
        if not blocked[r0 : r0 + box_h, c0 : c0 + box_w].any()
    ]
    if not candidates:
        return None
    return tuple(map(int, candidates[int(_RNG.integers(0, len(candidates)))]))


def _generate_once() -> np.ndarray | None:
    frame_count = int(_RNG.integers(3, 6))
    counts = sorted(map(int, _RNG.choice(np.arange(1, 10), size=frame_count, replace=False)))
    if counts == list(range(counts[0], counts[0] + frame_count)) and float(_RNG.random()) < 0.8:
        return None

    spiral_cells, (min_r, min_c, max_r, max_c) = _relative_spiral_bbox(counts)
    height = int(_RNG.integers(22, 31))
    width = int(_RNG.integers(22, 31))

    start_r_lo = 1 - min_r
    start_r_hi = height - 2 - max_r
    start_c_lo = 3 - min_c
    start_c_hi = width - 2 - max_c
    if start_r_lo > start_r_hi or start_c_lo > start_c_hi:
        return None

    start = (
        int(_RNG.integers(start_r_lo, start_r_hi + 1)),
        int(_RNG.integers(start_c_lo, start_c_hi + 1)),
    )

    palette_choices = [c for c in range(1, 10) if c != GREY]
    _RNG.shuffle(palette_choices)
    horizontal_color = int(palette_choices[0])
    vertical_color = int(palette_choices[1])
    start_color = int(palette_choices[2])
    frame_palette = palette_choices[3:]
    frame_colors = [frame_palette[i % len(frame_palette)] for i in range(frame_count)]

    grid = np.full((height, width), BG, dtype=int)
    grid[0, 0] = horizontal_color
    grid[0, 1] = vertical_color
    grid[start] = start_color

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[0, 0] = True
    occupied[0, 1] = True
    occupied[start] = True

    for dr, dc in spiral_cells[1:]:
        rr = start[0] + dr
        cc = start[1] + dc
        occupied[rr, cc] = True

    blocked = _dilate8(occupied)

    frame_specs = []
    for idx, count in enumerate(counts):
        dims = _feasible_frame_dims(count)
        if not dims:
            return None
        box_h, box_w = dims[int(_RNG.integers(0, len(dims)))]
        frame_specs.append(
            {
                "count": count,
                "color": int(frame_colors[idx]),
                "box_h": int(box_h),
                "box_w": int(box_w),
            }
        )

    frame_specs.sort(key=lambda spec: spec["box_h"] * spec["box_w"], reverse=True)
    for spec in frame_specs:
        placed = _place_box(blocked, spec["box_h"], spec["box_w"])
        if placed is None:
            return None
        r0, c0 = placed
        r1 = r0 + spec["box_h"] - 1
        c1 = c0 + spec["box_w"] - 1
        grid[r0 : r1 + 1, c0] = spec["color"]
        grid[r0 : r1 + 1, c1] = spec["color"]
        grid[r0, c0 : c1 + 1] = spec["color"]
        grid[r1, c0 : c1 + 1] = spec["color"]

        inner_h = spec["box_h"] - 2
        inner_w = spec["box_w"] - 2
        for mr, mc in _sample_marker_positions(inner_h, inner_w, spec["count"]):
            grid[r0 + 1 + mr, c0 + 1 + mc] = GREY

        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        blocked = _dilate8(occupied)

    if int(np.count_nonzero(grid == BG)) <= (grid.size // 2):
        return None

    expected = _render_output(
        grid.shape,
        BG,
        horizontal_color,
        vertical_color,
        start_color,
        start,
        counts,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-count scene")
