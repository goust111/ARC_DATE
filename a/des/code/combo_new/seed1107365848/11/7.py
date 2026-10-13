import random

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(points) != 2 * (h + w) - 4:
        return False
    point_set = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in point_set):
                return False
    return True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _extract_frames(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    non_bg = grid[grid != BACKGROUND]
    assert non_bg.size > 0, "expected a non-empty puzzle"
    colors, counts = np.unique(non_bg, return_counts=True)
    frame_color = int(colors[np.argmax(counts)])

    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == frame_color):
        if _is_rect_frame(comp):
            frames.append(_bbox(comp))

    assert len(frames) >= 2, "expected at least two frame layers"
    frames.sort(key=lambda box: (box[2] - box[0] + 1) * (box[3] - box[1] + 1), reverse=True)
    for outer, inner in zip(frames, frames[1:], strict=False):
        or0, oc0, or1, oc1 = outer
        ir0, ic0, ir1, ic1 = inner
        assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "frames must be nested"
    return frame_color, frames


def _layer_marker_color(
    grid: np.ndarray,
    frame_color: int,
    frames: list[tuple[int, int, int, int]],
    index: int,
) -> int:
    r0, c0, r1, c1 = frames[index]
    ring = np.zeros_like(grid, dtype=bool)
    ring[r0 + 1 : r1, c0 + 1 : c1] = True
    if index + 1 < len(frames):
        nr0, nc0, nr1, nc1 = frames[index + 1]
        ring[nr0 : nr1 + 1, nc0 : nc1 + 1] = False
    markers = np.argwhere(ring & (grid != BACKGROUND) & (grid != frame_color))
    assert len(markers) == 1, "each layer must have exactly one marker in its own interior ring"
    mr, mc = map(int, markers[0])
    return int(grid[mr, mc])


def _packed_canvas(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    outer_r0, _outer_c0, outer_r1, outer_c1 = frames[0]
    heights = [r1 - r0 + 1 for r0, _c0, r1, _c1 in frames]
    widths = [c1 - c0 + 1 for _r0, c0, _r1, c1 in frames]
    total_width = sum(widths) + len(widths) - 1
    outer_h = outer_r1 - outer_r0 + 1

    start_r = outer_r0
    start_c = outer_c1 + 1
    if start_r + outer_h <= shape[0] and start_c + total_width <= shape[1]:
        out = np.full(shape, BACKGROUND, dtype=int)
    else:
        out = np.full((outer_h, total_width), BACKGROUND, dtype=int)
        start_r = 0
        start_c = 0

    cursor = start_c
    for h, w, color in zip(heights, widths, colors, strict=False):
        _draw_frame(out, start_r, cursor, h, w, color)
        cursor += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a nested stack of grey rectangular frames into a left-to-right row of separately recolored frames.

    Input structure:
    - The background is black.
    - One connected arrangement on the left side contains several 1-cell-thick grey rectangular frames nested concentrically, with a 1-cell black gap between consecutive frames.
    - Each layer's own black interior ring contains exactly one non-black, non-grey marker pixel; the innermost frame's interior also contains exactly one such marker.
    - The area to the right of the outer frame is empty workspace.

    Rule:
    - Detect the grey frame components and order them from the outermost rectangle to the innermost rectangle.
    - For each layer, read the unique marker in that layer's own interior ring and use that marker's color to recolor the corresponding frame; the marker itself does not survive.
    - Clear the original nested structure to black and redraw the recolored frames side by side in outer-to-inner order, top-aligned and separated by one black column.

    Key insight:
    - All frames share the same visible grey color, so the layer identities come from geometric containment; each layer's isolated marker is only a color key for that one ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color, frames = _extract_frames(g)
    recolors = [_layer_marker_color(g, frame_color, frames, i) for i in range(len(frames))]
    return _packed_canvas(g.shape, frames, recolors)


def _ring_cells(
    frames: list[tuple[int, int, int, int]],
    index: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frames[index]
    cells: list[tuple[int, int]] = []
    if index + 1 < len(frames):
        nr0, nc0, nr1, nc1 = frames[index + 1]
    else:
        nr0 = nc0 = nr1 = nc1 = -1
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if index + 1 < len(frames) and nr0 <= r <= nr1 and nc0 <= c <= nc1:
                continue
            cells.append((int(r), int(c)))
    return cells


def _build_input(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
    marker_positions: list[tuple[int, int]],
    marker_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for r0, c0, r1, c1 in frames:
        _draw_frame(grid, r0, c0, r1 - r0 + 1, c1 - c0 + 1, FRAME_COLOR)
    for (r, c), color in zip(marker_positions, marker_colors, strict=False):
        grid[r, c] = int(color)
    return grid


def _expected_output(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
    marker_colors: list[int],
) -> np.ndarray:
    return _packed_canvas(shape, frames, marker_colors)


def _sample_case() -> tuple[np.ndarray, np.ndarray]:
    # Keeping the packed row on the same canvas under the ARC 30x30 cap
    # restricts generated cases to two layers, although the solver accepts
    # any number of layers that still fits.
    layer_count = 2
    outer_h = random.randint(7, 20)
    outer_w = random.randint(7, 10)
    inner_h = outer_h - 4
    inner_w = outer_w - 4
    packed_width = outer_w + inner_w + 1

    max_left = 30 - outer_w - packed_width
    assert max_left >= 0
    left_margin = random.randint(0, min(4, max_left))
    max_right = 30 - left_margin - outer_w - packed_width
    right_margin = random.randint(0, min(4, max_right))
    width = left_margin + outer_w + packed_width + right_margin
    assert left_margin + outer_w <= width // 2, "nested stack must remain on the left side"

    top_margin = random.randint(0, 6)
    bottom_margin = random.randint(0, 30 - outer_h - top_margin)
    height = top_margin + outer_h + bottom_margin

    frames = []
    for i in range(layer_count):
        r0 = top_margin + 2 * i
        c0 = left_margin + 2 * i
        h = outer_h - 4 * i
        w = outer_w - 4 * i
        frames.append((r0, c0, r0 + h - 1, c0 + w - 1))

    marker_positions = [random.choice(_ring_cells(frames, i)) for i in range(layer_count)]
    marker_colors = random.sample(MARKER_COLORS, layer_count)
    inp = _build_input((height, width), frames, marker_positions, marker_colors)
    out = _expected_output((height, width), frames, marker_colors)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, expected = _sample_case()
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
