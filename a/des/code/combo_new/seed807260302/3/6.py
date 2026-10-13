from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
INDICATOR_COLORS = (BLUE, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ORIENTATIONS = ("horizontal", "vertical")
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue: deque[tuple[int, int]] = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            pts.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(pts, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _frame_bboxes(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must have a usable interior"
        border = _border_cells(r0, c0, r1, c1)
        assert set(pts) == border, "grey component must be a complete one-pixel rectangular frame"
        assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] != GREY), "frame interior must not contain grey"
        frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one grey frame"
    return frames


def _find_indicator(interior: np.ndarray) -> tuple[int, str, np.ndarray]:
    marker_mask = (interior != BLACK) & (interior != RED) & (interior != GREY)
    coords = np.argwhere(marker_mask)
    assert coords.size, "each frame must contain an indicator"

    colors = {int(interior[r, c]) for r, c in coords}
    assert len(colors) == 1, "each frame must contain exactly one indicator color"
    color = next(iter(colors))

    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert int(marker_mask.sum()) == height * width, "indicator must be a solid bar"

    if height == 1 and width in (2, 3):
        orient = "horizontal"
    elif width == 1 and height in (2, 3):
        orient = "vertical"
    else:
        raise AssertionError("indicator must be a short horizontal or vertical bar")

    return int(color), orient, coords.astype(int)


def _reflect_coords(coords: np.ndarray, interior_shape: tuple[int, int], orient: str) -> np.ndarray:
    h, w = interior_shape
    reflected = coords.copy()
    if orient == "horizontal":
        reflected[:, 0] = h - 1 - reflected[:, 0]
    elif orient == "vertical":
        reflected[:, 1] = w - 1 - reflected[:, 1]
    else:
        raise AssertionError(f"unknown orientation {orient}")
    return reflected


def _paint_frame(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each framed red pattern according to the enclosed bar and recolor that frame with the bar color.

    Input structure:
    - The background is black.
    - Several separated one-pixel-thick grey rectangular frames enclose a red
      asymmetric pattern and one short non-red, non-grey indicator bar.
    - Each indicator bar is either horizontal with length two or three, or
      vertical with length two or three, and each frame uses a different
      indicator color.

    Rule:
    - For every grey frame, inspect the indicator bar inside its interior.
    - A horizontal indicator reflects the red pixels across the interior's
      horizontal middle axis, moving them up/down; a vertical indicator reflects
      the red pixels across the vertical middle axis, moving them left/right.
    - Clear the indicator from the interior and recolor the entire grey border
      with the indicator's color while keeping reflected pixels red.

    Key insight:
    The small bar carries two independent instructions: its geometry names the
    mirror axis, and its color names the replacement border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    out = g.copy()
    for box in _frame_bboxes(g):
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        indicator_color, orient, _indicator_coords = _find_indicator(interior)
        red_coords = np.argwhere(interior == RED)

        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
        if red_coords.size:
            reflected = _reflect_coords(red_coords.astype(int), interior.shape, orient)
            rr = reflected[:, 0] + r0 + 1
            cc = reflected[:, 1] + c0 + 1
            out[rr, cc] = RED
        _paint_frame(out, box, indicator_color)

    return out


def _indicator_cells(
    ih: int, iw: int, orient: str, corner: str, length: int
) -> list[tuple[int, int]]:
    top = "top" in corner
    left = "left" in corner
    if orient == "horizontal":
        r = 0 if top else ih - 1
        start = 0 if left else iw - length
        return [(r, c) for c in range(start, start + length)]

    c = 0 if left else iw - 1
    start = 0 if top else ih - length
    return [(r, c) for r in range(start, start + length)]


def _mask_from_cells(shape: tuple[int, int], cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _expanded_forbidden(
    shape: tuple[int, int], cells: list[tuple[int, int]], pad: int
) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros(shape, dtype=bool)
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                forbidden[rr, cc] = True
    return forbidden


def _random_connected_mask(allowed: np.ndarray, target: int) -> np.ndarray | None:
    choices = [(int(r), int(c)) for r, c in np.argwhere(allowed)]
    if len(choices) < target:
        return None

    start = random.choice(choices)
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < allowed.shape[0]
                    and 0 <= nc < allowed.shape[1]
                    and allowed[nr, nc]
                    and (nr, nc) not in cells
                ):
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros(allowed.shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _irregular(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.shape[0] < 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1 or c0 == c1:
        return False
    return coords.shape[0] < (r1 - r0 + 1) * (c1 - c0 + 1)


def _reflect_mask(mask: np.ndarray, orient: str) -> np.ndarray:
    return np.flipud(mask) if orient == "horizontal" else np.fliplr(mask)


def _make_red_mask(
    ih: int, iw: int, orient: str, indicator_cells: list[tuple[int, int]]
) -> np.ndarray | None:
    indicator = _mask_from_cells((ih, iw), indicator_cells)
    forbidden = _expanded_forbidden((ih, iw), indicator_cells, pad=1)
    reflected_indicator = _reflect_mask(indicator, orient)
    allowed = ~(forbidden | reflected_indicator)

    if int(allowed.sum()) < 4:
        return None
    for _ in range(100):
        target = random.randint(4, min(10, int(allowed.sum())))
        mask = _random_connected_mask(allowed, target)
        if mask is None:
            continue
        flipped = _reflect_mask(mask, orient)
        if not _irregular(mask):
            continue
        if np.array_equal(mask, flipped):
            continue
        if np.any(flipped & indicator):
            continue
        return mask
    return None


def _free_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rows, cols = occupied.shape
    rr0, rr1 = max(0, r0 - pad), min(rows, r0 + h + pad)
    cc0, cc1 = max(0, c0 - pad), min(cols, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_frame(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    orient: str,
) -> bool:
    h, w = grid.shape
    for _ in range(160):
        fh = random.randint(7, min(11, h - 2))
        fw = random.randint(7, min(11, w - 2))
        r0 = random.randint(1, h - fh - 1)
        c0 = random.randint(1, w - fw - 1)
        if not _free_box(occupied, r0, c0, fh, fw, pad=1):
            continue

        ih, iw = fh - 2, fw - 2
        length = random.randint(2, 3)
        corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
        cells = _indicator_cells(ih, iw, orient, corner, length)
        red = _make_red_mask(ih, iw, orient, cells)
        if red is None:
            continue

        r1, c1 = r0 + fh - 1, c0 + fw - 1
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        interior[red] = RED
        for rr, cc in cells:
            interior[rr, cc] = color

        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 25)
    w = random.randint(15, 25)
    count = random.randint(2, 4)

    colors = list(INDICATOR_COLORS)
    random.shuffle(colors)
    colors = colors[:count]

    orientations = ["horizontal", "vertical"]
    while len(orientations) < count:
        orientations.append(random.choice(ORIENTATIONS))
    random.shuffle(orientations)

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for color, orient in zip(colors, orientations):
        if not _place_frame(grid, occupied, int(color), orient):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed807260302/3/6")
