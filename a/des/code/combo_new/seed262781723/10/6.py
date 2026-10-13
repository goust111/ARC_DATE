from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
CANVAS = 30
COPY_COUNTS = (2, 3)
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 60


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _marker_orientation(markers: np.ndarray) -> str:
    rows = np.unique(markers[:, 0])
    cols = np.unique(markers[:, 1])
    if len(rows) == 1 and len(cols) >= 2:
        ordered = np.sort(cols)
        assert np.array_equal(ordered, np.arange(int(ordered[0]), int(ordered[-1]) + 1))
        return "horizontal"
    if len(cols) == 1 and len(rows) >= 2:
        ordered = np.sort(rows)
        assert np.array_equal(ordered, np.arange(int(ordered[0]), int(ordered[-1]) + 1))
        return "vertical"
    raise AssertionError("instruction pixels must form a straight contiguous line")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack extra copies of each marked shape in the direction encoded by its yellow instruction line.

    Input structure:
    - The background is black.
    - Each non-background 4-connected component is one shape made of a single base color
      plus some embedded yellow instruction pixels.
    - Within a shape, the yellow pixels form one straight horizontal or vertical line.

    Rule:
    - For each shape, count its yellow instruction pixels; that count N is the number of
      extra copies to add, so the output contains the original plus N adjacent copies.
    - The instruction line's orientation gives the stacking axis, and the line's bias toward
      the left/right or top/bottom side of the shape chooses the stacking direction.
    - Each copy is translated by exactly one shape bounding-box width or height per step.
      Copies keep the shape geometry but replace the yellow instruction pixels with the
      shape's base color, while the original shape stays unchanged.

    Key insight:
    - The yellow line carries two independent signals at once: its length is the copy count,
      and its placement inside the shape determines both the stacking axis and the facing side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    components = _components(g != BG)
    assert components, "expected at least one shape"

    for coords in components:
        colors = {int(g[r, c]) for r, c in coords}
        assert YELLOW in colors, "each shape must contain yellow instruction pixels"
        base_colors = colors - {YELLOW}
        assert len(base_colors) == 1, "each shape must have exactly one non-yellow base color"
        base_color = next(iter(base_colors))

        marker_mask = np.array([int(g[r, c]) == YELLOW for r, c in coords], dtype=bool)
        markers = coords[marker_mask]
        assert len(markers) >= 2, "instruction line must have a clear orientation"
        copies = int(len(markers))
        orientation = _marker_orientation(markers)

        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        local = coords - np.array([r0, c0], dtype=int)

        if orientation == "horizontal":
            left_gap = int(markers[:, 1].min()) - c0
            right_gap = c1 - int(markers[:, 1].max())
            assert left_gap != right_gap, "horizontal markers must be biased left or right"
            dr = 0
            dc = -width if left_gap < right_gap else width
        else:
            top_gap = int(markers[:, 0].min()) - r0
            bottom_gap = r1 - int(markers[:, 0].max())
            assert top_gap != bottom_gap, "vertical markers must be biased up or down"
            dr = -height if top_gap < bottom_gap else height
            dc = 0

        for step in range(1, copies + 1):
            rr = r0 + local[:, 0] + step * dr
            cc = c0 + local[:, 1] + step * dc
            assert np.all((0 <= rr) & (rr < g.shape[0]))
            assert np.all((0 <= cc) & (cc < g.shape[1]))
            assert np.all(out[rr, cc] == BG), "copies must fit into empty space"
            out[rr, cc] = int(base_color)

    return out


def _line_cells(height: int, width: int, direction: str, count: int) -> np.ndarray:
    if direction in ("left", "right"):
        row = random.randint(1, height - 2)
        slack = width - count
        starts: list[int] = []
        for left_gap in range(1, slack):
            right_gap = slack - left_gap
            if right_gap < 1:
                continue
            if direction == "left" and left_gap < right_gap:
                starts.append(left_gap)
            if direction == "right" and right_gap < left_gap:
                starts.append(left_gap)
        assert starts
        start = random.choice(starts)
        return np.array([(row, c) for c in range(start, start + count)], dtype=int)

    col = random.randint(1, width - 2)
    slack = height - count
    starts = []
    for top_gap in range(1, slack):
        bottom_gap = slack - top_gap
        if bottom_gap < 1:
            continue
        if direction == "up" and top_gap < bottom_gap:
            starts.append(top_gap)
        if direction == "down" and bottom_gap < top_gap:
            starts.append(top_gap)
    assert starts
    start = random.choice(starts)
    return np.array([(r, col) for r in range(start, start + count)], dtype=int)


def _random_cut_cells(height: int, width: int) -> list[tuple[int, int]]:
    mode = random.choice(("corner", "corner", "notch"))
    cells: list[tuple[int, int]] = []

    if mode == "corner":
        dh = random.randint(1, min(2, height - 2))
        dw = random.randint(1, min(2, width - 2))
        corner = random.choice(("tl", "tr", "bl", "br"))
        row_range = range(dh) if "t" in corner else range(height - dh, height)
        col_range = range(dw) if "l" in corner else range(width - dw, width)
        cells = [(r, c) for r in row_range for c in col_range]
    elif random.random() < 0.5:
        depth = random.randint(1, min(2, height - 2))
        notch_w = random.randint(1, width - 2)
        start = random.randint(1, width - notch_w - 1)
        edge = random.choice(("top", "bottom"))
        row_range = range(depth) if edge == "top" else range(height - depth, height)
        cells = [(r, c) for r in row_range for c in range(start, start + notch_w)]
    else:
        depth = random.randint(1, min(2, width - 2))
        notch_h = random.randint(1, height - 2)
        start = random.randint(1, height - notch_h - 1)
        edge = random.choice(("left", "right"))
        col_range = range(depth) if edge == "left" else range(width - depth, width)
        cells = [(r, c) for r in range(start, start + notch_h) for c in col_range]

    return cells


def _shape_mask(height: int, width: int, markers: np.ndarray) -> np.ndarray:
    protected = {tuple(map(int, cell)) for cell in markers}
    mask = np.ones((height, width), dtype=bool)

    for _ in range(random.randint(0, 3)):
        candidate = mask.copy()
        cells = _random_cut_cells(height, width)
        if any(cell in protected for cell in cells):
            continue
        for r, c in cells:
            candidate[r, c] = False
        if int(candidate.sum()) <= len(protected):
            continue
        if not _touches_all_edges(candidate):
            continue
        if len(_components(candidate)) != 1:
            continue
        mask = candidate

    assert np.all(mask[markers[:, 0], markers[:, 1]])
    assert _touches_all_edges(mask)
    assert len(_components(mask)) == 1
    return mask


def _sample_shape(direction: str, count: int, color: int) -> dict[str, object]:
    axis_min = count + 3
    if direction in ("left", "right"):
        height = random.randint(4, 7)
        width = random.randint(axis_min, 7)
    else:
        height = random.randint(axis_min, 7)
        width = random.randint(4, 7)

    markers = _line_cells(height, width, direction, count)
    mask = _shape_mask(height, width, markers)

    if direction == "left":
        foot_h, foot_w = height, (count + 1) * width
        input_offset = (0, count * width)
    elif direction == "right":
        foot_h, foot_w = height, (count + 1) * width
        input_offset = (0, 0)
    elif direction == "up":
        foot_h, foot_w = (count + 1) * height, width
        input_offset = (count * height, 0)
    else:
        foot_h, foot_w = (count + 1) * height, width
        input_offset = (0, 0)

    return {
        "direction": direction,
        "count": count,
        "color": color,
        "mask": mask,
        "markers": markers,
        "height": height,
        "width": width,
        "foot_h": foot_h,
        "foot_w": foot_w,
        "input_offset": input_offset,
    }


def _find_slot(occupied: np.ndarray, foot_h: int, foot_w: int) -> tuple[int, int] | None:
    height, width = occupied.shape
    choices: list[tuple[int, int]] = []
    for r0 in range(height - foot_h + 1):
        for c0 in range(width - foot_w + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + foot_h + 1)
            cc1 = min(width, c0 + foot_w + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                choices.append((r0, c0))
    if not choices:
        return None
    return random.choice(choices)


def _paint_shape(grid: np.ndarray, spec: dict[str, object], foot_r: int, foot_c: int) -> None:
    mask = np.asarray(spec["mask"], dtype=bool)
    markers = np.asarray(spec["markers"], dtype=int)
    color = int(spec["color"])
    off_r, off_c = map(int, spec["input_offset"])
    r0 = foot_r + off_r
    c0 = foot_c + off_c

    rows, cols = np.where(mask)
    grid[r0 + rows, c0 + cols] = color
    grid[r0 + markers[:, 0], c0 + markers[:, 1]] = YELLOW


def _crop_to_output_window(grid: np.ndarray, footprints: list[tuple[int, int, int, int]]) -> np.ndarray:
    top = min(r0 for r0, _c0, _h, _w in footprints)
    left = min(c0 for _r0, c0, _h, _w in footprints)
    bottom = max(r0 + h - 1 for r0, _c0, h, _w in footprints)
    right = max(c0 + w - 1 for _r0, c0, _h, w in footprints)

    crop_top = random.randint(0, top)
    crop_left = random.randint(0, left)
    crop_bottom = random.randint(bottom, CANVAS - 1)
    crop_right = random.randint(right, CANVAS - 1)
    return grid[crop_top : crop_bottom + 1, crop_left : crop_right + 1].copy()


def _scene_attempt() -> np.ndarray | None:
    n_shapes = random.randint(1, 3)
    directions = random.sample(("left", "right", "up", "down"), k=n_shapes)
    colors = random.sample(PALETTE, k=n_shapes)
    specs: list[dict[str, object]] = []
    for direction, color in zip(directions, colors, strict=False):
        count = random.choice(COPY_COUNTS)
        specs.append(_sample_shape(direction, count, color))

    occupied = np.zeros((CANVAS, CANVAS), dtype=bool)
    grid = np.full((CANVAS, CANVAS), BG, dtype=int)
    footprints: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda item: int(item["foot_h"]) * int(item["foot_w"]), reverse=True):
        placed = None
        foot_h = int(spec["foot_h"])
        foot_w = int(spec["foot_w"])
        for _ in range(PLACEMENT_ATTEMPTS):
            placed = _find_slot(occupied, foot_h, foot_w)
            if placed is not None:
                break
        if placed is None:
            return None
        foot_r, foot_c = placed
        occupied[foot_r : foot_r + foot_h, foot_c : foot_c + foot_w] = True
        footprints.append((foot_r, foot_c, foot_h, foot_w))
        _paint_shape(grid, spec, foot_r, foot_c)

    cropped = _crop_to_output_window(grid, footprints)
    transformed = main(cropped)
    assert transformed.shape == cropped.shape
    assert np.count_nonzero(cropped == YELLOW) > 0
    assert np.count_nonzero(transformed) > np.count_nonzero(cropped)
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
