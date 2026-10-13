import random
from collections import deque

import numpy as np


SIDE_ORDER = ("top", "right", "bottom", "left")
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touching_sides(coords: list[tuple[int, int]], h: int, w: int) -> set[str]:
    sides = set()
    for r, c in coords:
        if r == 0:
            sides.add("top")
        if r == h - 1:
            sides.add("bottom")
        if c == 0:
            sides.add("left")
        if c == w - 1:
            sides.add("right")
    return sides


def _assert_arrow_shape(coords: list[tuple[int, int]], side: str, shape: tuple[int, int]) -> None:
    h, w = shape
    pts = set(coords)
    assert len(pts) == 4, "arrow marker must have exactly four pixels"
    if side == "top":
        border = [(r, c) for r, c in pts if r == 0]
        assert len(border) == 3
        cols = sorted(c for _, c in border)
        mid = cols[1]
        assert cols == [mid - 1, mid, mid + 1]
        assert (1, mid) in pts
    elif side == "bottom":
        border = [(r, c) for r, c in pts if r == h - 1]
        assert len(border) == 3
        cols = sorted(c for _, c in border)
        mid = cols[1]
        assert cols == [mid - 1, mid, mid + 1]
        assert (h - 2, mid) in pts
    elif side == "left":
        border = [(r, c) for r, c in pts if c == 0]
        assert len(border) == 3
        rows = sorted(r for r, _ in border)
        mid = rows[1]
        assert rows == [mid - 1, mid, mid + 1]
        assert (mid, 1) in pts
    else:
        border = [(r, c) for r, c in pts if c == w - 1]
        assert len(border) == 3
        rows = sorted(r for r, _ in border)
        mid = rows[1]
        assert rows == [mid - 1, mid, mid + 1]
        assert (mid, w - 2) in pts


def _analyze_frame(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    bg: int,
) -> tuple[int, int, int, int, str, tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3, "frame bbox must have an interior"

    local = {(r - r0, c - c0) for r, c in coords}
    border = set()
    for c in range(width):
        border.add((0, c))
        border.add((height - 1, c))
    for r in range(height):
        border.add((r, 0))
        border.add((r, width - 1))

    missing = border - local
    extra = local - border
    assert not extra, "frame pixels must lie exactly on a rectangle border"
    assert len(missing) == 1, "frame border must contain exactly one gap"

    gap_r, gap_c = next(iter(missing))
    assert (gap_r, gap_c) not in {
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
    }, "gap must sit on one side, not in a corner"

    if height > 2 and width > 2:
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert np.all(interior == bg), "frame interior must start as background"

    if gap_r == 0:
        side = "top"
    elif gap_r == height - 1:
        side = "bottom"
    elif gap_c == 0:
        side = "left"
    else:
        side = "right"
    return r0, r1, c0, c1, side, (r0 + gap_r, c0 + gap_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular frame using the color of the edge arrow on the same side as the gap.

    Input structure:
    - The background is black or otherwise the dominant color.
    - Four small monochrome arrow markers sit on the middle of the top, right, bottom, and left grid edges.
    - Inside the grid are one or more single-color rectangular frames whose borders are one pixel thick and each contain exactly one one-pixel gap on one side.

    Rule:
    - Read the edge arrows as a side-to-color legend, using the arrow's border position rather than the direction it points.
    - For every internal frame, determine which side contains its unique border gap.
    - Fill the entire interior of that frame with the legend color for the matching side, and also paint the gap pixel with that same color.
    - Leave the original frame border, the arrow markers, and all exterior background pixels unchanged.

    Key insight:
    - The arrows are not movement cues; they are four color labels attached to the four border directions, and each frame inherits its fill color from its missing side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)
    out = g.copy()

    side_to_color: dict[str, int] = {}
    frames: list[tuple[int, int, int, int, str, tuple[int, int]]] = []

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        for comp in _components(g == color):
            sides = _touching_sides(comp, h, w)
            if sides:
                assert len(sides) == 1, "each edge marker must touch exactly one border side"
                side = next(iter(sides))
                _assert_arrow_shape(comp, side, g.shape)
                assert side not in side_to_color, "expected exactly one arrow marker per side"
                side_to_color[side] = color
            else:
                frames.append(_analyze_frame(g, comp, bg))

    assert set(side_to_color) == set(SIDE_ORDER), "all four edge arrows must be present"
    assert len(set(side_to_color.values())) == 4, "edge arrow colors must be distinct"
    assert frames, "expected at least one internal gapped frame"

    for r0, r1, c0, c1, side, gap_cell in frames:
        fill_color = side_to_color[side]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
        out[gap_cell] = fill_color

    return out


BG = 0
FRAME_COLOR = 5
ARROW_COLOR_POOL = tuple(c for c in range(1, 10) if c != FRAME_COLOR)
GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
FRAME_COUNT_RANGE = (1, 5)
FRAME_H_RANGE = (4, 10)
FRAME_W_RANGE = (4, 10)
INNER_MARGIN = 3
FRAME_PADDING = 1
PLACEMENT_TRIES = 120
GEN_TRIES = 400


def _draw_arrows(grid: np.ndarray, side_to_color: dict[str, int]) -> None:
    h, w = grid.shape
    mid_r = h // 2
    mid_c = w // 2

    color = side_to_color["top"]
    grid[0, mid_c - 1 : mid_c + 2] = color
    grid[1, mid_c] = color

    color = side_to_color["bottom"]
    grid[h - 1, mid_c - 1 : mid_c + 2] = color
    grid[h - 2, mid_c] = color

    color = side_to_color["left"]
    grid[mid_r - 1 : mid_r + 2, 0] = color
    grid[mid_r, 1] = color

    color = side_to_color["right"]
    grid[mid_r - 1 : mid_r + 2, w - 1] = color
    grid[mid_r, w - 2] = color


def _sample_gap_sides(count: int) -> list[str]:
    sides: list[str] = []
    while len(sides) < count:
        batch = list(SIDE_ORDER)
        random.shuffle(batch)
        take = min(count - len(sides), len(batch))
        sides.extend(batch[:take])
    return sides


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and ar1 >= br0 and ac0 <= bc1 and ac1 >= bc0


def _frame_gap_cell(box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    return random.randint(r0 + 1, r1 - 1), c1


def _draw_frame(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    gap_cell: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[gap_cell] = BG


def _paint_expected(
    grid: np.ndarray,
    frames: list[tuple[tuple[int, int, int, int], str, tuple[int, int]]],
    side_to_color: dict[str, int],
) -> np.ndarray:
    out = grid.copy()
    for box, side, gap_cell in frames:
        r0, c0, r1, c1 = box
        color = side_to_color[side]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
        out[gap_cell] = color
    return out


def _place_frames(h: int, w: int, count: int) -> list[tuple[tuple[int, int, int, int], str, tuple[int, int]]] | None:
    max_h = min(FRAME_H_RANGE[1], h - 2 * INNER_MARGIN)
    max_w = min(FRAME_W_RANGE[1], w - 2 * INNER_MARGIN)
    if max_h < FRAME_H_RANGE[0] or max_w < FRAME_W_RANGE[0]:
        return None

    placed_boxes: list[tuple[int, int, int, int]] = []
    frames: list[tuple[tuple[int, int, int, int], str, tuple[int, int]]] = []
    for side in _sample_gap_sides(count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            fh = random.randint(FRAME_H_RANGE[0], max_h)
            fw = random.randint(FRAME_W_RANGE[0], max_w)
            r0 = random.randint(INNER_MARGIN, h - INNER_MARGIN - fh)
            c0 = random.randint(INNER_MARGIN, w - INNER_MARGIN - fw)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            expanded = (box[0] - FRAME_PADDING, box[1] - FRAME_PADDING, box[2] + FRAME_PADDING, box[3] + FRAME_PADDING)
            if any(_boxes_overlap(expanded, prev) for prev in placed_boxes):
                continue
            gap_cell = _frame_gap_cell(box, side)
            placed_boxes.append(expanded)
            frames.append((box, side, gap_cell))
            placed = True
            break
        if not placed:
            return None
    return frames


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    side_colors = random.sample(ARROW_COLOR_POOL, 4)
    side_to_color = dict(zip(SIDE_ORDER, side_colors))
    _draw_arrows(grid, side_to_color)

    frame_count = random.randint(*FRAME_COUNT_RANGE)
    frames = _place_frames(h, w, frame_count)
    if frames is None:
        return None

    for box, _, gap_cell in frames:
        _draw_frame(grid, box, gap_cell)

    expected = _paint_expected(grid, frames, side_to_color)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_TRIES} attempts")
