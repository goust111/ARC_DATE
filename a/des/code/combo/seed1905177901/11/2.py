from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
BG_COLOR = 8
GEN_RETRIES = 400
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
OBJECT_COUNT_RANGE = (2, 5)
RECT_HEIGHT_RANGE = (2, 5)
RECT_WIDTH_RANGE = (2, 6)
MARKER_ATTEMPTS = 120
RECT_ATTEMPTS = 160


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, "expected a unique dominant background color"
    return int(winners[0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _parse_colored_object(
    grid: np.ndarray, color: int, bg: int
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    del bg  # solver only needs it for the external color split
    h, w = grid.shape
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "every non-background color must appear"

    border_mask = (
        (coords[:, 0] == 0)
        | (coords[:, 0] == h - 1)
        | (coords[:, 1] == 0)
        | (coords[:, 1] == w - 1)
    )
    marker_coords = coords[border_mask]
    rect_coords = coords[~border_mask]

    assert marker_coords.shape == (1, 2), "expected one border marker per color"
    assert rect_coords.size > 0, "expected one interior rectangle per color"

    marker = tuple(map(int, marker_coords[0]))
    mr, mc = marker
    on_horizontal = mr in (0, h - 1)
    on_vertical = mc in (0, w - 1)
    assert on_horizontal ^ on_vertical, "marker must lie on exactly one edge"

    r0, c0, r1, c1 = _bbox(rect_coords)
    rect_h = int(r1 - r0 + 1)
    rect_w = int(c1 - c0 + 1)
    assert rect_coords.shape[0] == rect_h * rect_w, "object body must be a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)), "rectangle must be dense"
    assert 0 < r0 and r1 < h - 1 and 0 < c0 and c1 < w - 1, "rectangle must stay off the boundary"

    return marker, (r0, c0, r1, c1)


def _draw_transformed(
    out: np.ndarray, color: int, marker: tuple[int, int], bbox: tuple[int, int, int, int]
) -> None:
    h, w = out.shape
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    rect_h = int(r1 - r0 + 1)
    rect_w = int(c1 - c0 + 1)

    if mr == 0:
        assert 0 < mc < w - 1
        out[:rect_h, mc] = int(color)
    elif mr == h - 1:
        assert 0 < mc < w - 1
        out[h - rect_h :, mc] = int(color)
    elif mc == 0:
        assert 0 < mr < h - 1
        out[mr, :rect_w] = int(color)
    else:
        assert mc == w - 1
        assert 0 < mr < h - 1
        out[mr, w - rect_w :] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each interior rectangle to its same-colored border marker and compress it into an edge-anchored line.

    Input structure:
    - Background is dominant.
    - Each non-background color appears as one single border marker plus one solid interior rectangle.

    Rule:
    - The marker edge determines the movement direction.
    - The rectangle is moved until it touches that edge.
    - It keeps its thickness along the movement direction, but is compressed to width 1 in the perpendicular direction.
    - Its remaining free coordinate is aligned to the marker row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)

    out = np.full_like(g, int(bg))
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        marker, bbox = _parse_colored_object(g, color, bg)
        _draw_transformed(out, color, marker, bbox)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _sample_sides(n_objects: int) -> list[str]:
    sides: list[str] = []
    while len(sides) < n_objects:
        block = list(SIDES)
        random.shuffle(block)
        take = min(len(block), n_objects - len(sides))
        sides.extend(block[:take])
    random.shuffle(sides)
    if len(set(sides)) == 1 and n_objects > 1:
        choices = [side for side in SIDES if side != sides[0]]
        sides[-1] = random.choice(choices)
    return sides


def _sample_dims(grid_h: int, grid_w: int) -> tuple[int, int] | None:
    max_h = min(RECT_HEIGHT_RANGE[1], grid_h - 2)
    max_w = min(RECT_WIDTH_RANGE[1], grid_w - 2)
    if max_h < RECT_HEIGHT_RANGE[0] or max_w < RECT_WIDTH_RANGE[0]:
        return None
    return (
        random.randint(RECT_HEIGHT_RANGE[0], max_h),
        random.randint(RECT_WIDTH_RANGE[0], max_w),
    )


def _line_cells(
    side: str, grid_h: int, grid_w: int, anchor: int, rect_h: int, rect_w: int
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if side == "top":
        marker = (0, anchor)
        cells = [(r, anchor) for r in range(rect_h)]
    elif side == "bottom":
        marker = (grid_h - 1, anchor)
        cells = [(r, anchor) for r in range(grid_h - rect_h, grid_h)]
    elif side == "left":
        marker = (anchor, 0)
        cells = [(anchor, c) for c in range(rect_w)]
    else:
        assert side == "right"
        marker = (anchor, grid_w - 1)
        cells = [(anchor, c) for c in range(grid_w - rect_w, grid_w)]
    return marker, cells


def _place_marker_line(
    occupied: np.ndarray, side: str, rect_h: int, rect_w: int
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    grid_h, grid_w = occupied.shape
    for _ in range(MARKER_ATTEMPTS):
        if side in ("top", "bottom"):
            anchor = random.randint(1, grid_w - 2)
        else:
            anchor = random.randint(1, grid_h - 2)

        marker, cells = _line_cells(side, grid_h, grid_w, anchor, rect_h, rect_w)
        if any(occupied[r, c] for r, c in cells):
            continue

        for r, c in cells:
            occupied[r, c] = True
        return marker, cells
    return None


def _bbox_touches_marker(
    bbox: tuple[int, int, int, int], marker: tuple[int, int], grid_h: int, grid_w: int
) -> bool:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    if mr == 0:
        return r0 == 1 and c0 <= mc <= c1
    if mr == grid_h - 1:
        return r1 == grid_h - 2 and c0 <= mc <= c1
    if mc == 0:
        return c0 == 1 and r0 <= mr <= r1
    assert mc == grid_w - 1
    return c1 == grid_w - 2 and r0 <= mr <= r1


def _place_rectangle(
    occupied: np.ndarray,
    rect_h: int,
    rect_w: int,
    markers: list[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    grid_h, grid_w = occupied.shape
    dilated = _dilate8(occupied)

    for _ in range(RECT_ATTEMPTS):
        r0 = random.randint(1, grid_h - rect_h - 1)
        c0 = random.randint(1, grid_w - rect_w - 1)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        bbox = (r0, c0, r1, c1)

        if dilated[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        if any(_bbox_touches_marker(bbox, marker, grid_h, grid_w) for marker in markers):
            continue

        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        return bbox
    return None


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)

    colors = random.sample([c for c in range(10) if c != BG_COLOR], n_objects)
    sides = _sample_sides(n_objects)

    objects: list[dict] = []
    output_occupied = np.zeros((grid_h, grid_w), dtype=bool)
    expected = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    for color, side in zip(colors, sides):
        dims = _sample_dims(grid_h, grid_w)
        if dims is None:
            return None
        rect_h, rect_w = dims
        placed_line = _place_marker_line(output_occupied, side, rect_h, rect_w)
        if placed_line is None:
            return None
        marker, line_cells = placed_line
        for r, c in line_cells:
            expected[r, c] = int(color)
        objects.append(
            {
                "color": int(color),
                "side": side,
                "rect_h": int(rect_h),
                "rect_w": int(rect_w),
                "marker": marker,
            }
        )

    markers = [obj["marker"] for obj in objects]
    input_occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for obj in sorted(objects, key=lambda item: item["rect_h"] * item["rect_w"], reverse=True):
        bbox = _place_rectangle(input_occupied, obj["rect_h"], obj["rect_w"], markers)
        if bbox is None:
            return None
        obj["bbox"] = bbox

    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = obj["color"]
        mr, mc = obj["marker"]
        inp[mr, mc] = obj["color"]

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
