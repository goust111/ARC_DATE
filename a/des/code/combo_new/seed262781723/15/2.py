import random
from collections import Counter

import numpy as np


BG = 0
FRAME_COLOR = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    banned = set() if exclude is None else set(exclude)
    return [int(v) for v in np.unique(grid) if int(v) not in banned]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    parts: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(pts)
    return parts


def bbox_coords(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _find_frames(grid: np.ndarray) -> list[dict[str, tuple[int, int, int, int] | int]]:
    frames: list[dict[str, tuple[int, int, int, int] | int]] = []
    for pts in components_mask(grid == FRAME_COLOR, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "red frame must enclose a non-empty interior"

        local = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            local[r - r0, c - c0] = True

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(local, border), "red components must be hollow rectangles"

        inner = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": inner,
                "area": int((inner[2] - inner[0] + 1) * (inner[3] - inner[1] + 1)),
            }
        )

    areas = [int(frame["area"]) for frame in frames]
    assert frames and len(set(areas)) == len(areas), "frame interior areas must be unique"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each same-colored fragment group into the uniquely area-matched red frame interior.

    Input structure:
    - The background is black.
    - One color forms several red one-pixel-thick hollow rectangles whose interior areas are all different.
    - Every other non-black color marks one fragment group; pixels of the same color belong to the same group and stay away from the red frames.

    Rule:
    - Compute the interior area of every red hollow rectangle.
    - For each non-red fragment color, count its pixels and match it to the unique frame with the same interior area.
    - Take that color's overall bounding box, align its top-left corner to the matched frame interior's top-left corner, and translate all pixels by that same offset.
    - Clear the original fragment pixels and keep the red frame outlines unchanged.

    Key insight:
    - Pixel count solves the frame-to-group pairing, while the translated relative coordinates inside each color group determine where the moved pixels land.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    frames = _find_frames(g)
    frame_by_area = {int(frame["area"]): frame for frame in frames}

    out = g.copy()
    fragment_colors = colors_present(g, exclude={bg, FRAME_COLOR})
    fragment_mask = np.isin(g, fragment_colors)
    out[fragment_mask] = bg

    for color in fragment_colors:
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        assert pts, "every fragment color must appear"
        area = len(pts)
        assert area in frame_by_area, "fragment size must match a frame interior area"
        frame = frame_by_area[area]
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        gr0, gc0, gr1, gc1 = bbox_coords(pts)
        shift_r = int(ir0) - gr0
        shift_c = int(ic0) - gc0

        for r, c in pts:
            nr = r + shift_r
            nc = c + shift_c
            assert int(ir0) <= nr <= int(ir1) and int(ic0) <= nc <= int(ic1), (
                "translated fragment pixels must stay inside the matched frame interior"
            )
            out[nr, nc] = color

    return out


AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(1, 4), (2, 2), (4, 1)],
    5: [(1, 5), (5, 1)],
    6: [(1, 6), (2, 3), (3, 2), (6, 1)],
    7: [(1, 7), (7, 1)],
    8: [(1, 8), (2, 4), (4, 2), (8, 1)],
    9: [(1, 9), (3, 3), (9, 1)],
    10: [(2, 5), (5, 2)],
    12: [(2, 6), (3, 4), (4, 3), (6, 2)],
}
PAIR_COUNT_CHOICES = [2, 3, 3, 4]
GRID_SIZE_RANGE = (18, 30)
COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]
PLACEMENT_MARGIN = 1
PLACEMENT_TRIES = 200
GEN_RETRIES = 300


def _place_box(
    rng: random.Random,
    reserved: np.ndarray,
    box_h: int,
    box_w: int,
    margin: int,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    if box_h + 2 > h or box_w + 2 > w:
        return None

    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randint(1, h - box_h - 1)
        c0 = rng.randint(1, w - box_w - 1)
        rr0 = max(0, r0 - margin)
        cc0 = max(0, c0 - margin)
        rr1 = min(h, r0 + box_h + margin)
        cc1 = min(w, c0 + box_w + margin)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        reserved[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top : top + outer_h, left] = FRAME_COLOR
    grid[top : top + outer_h, left + outer_w - 1] = FRAME_COLOR
    grid[top, left : left + outer_w] = FRAME_COLOR
    grid[top + outer_h - 1, left : left + outer_w] = FRAME_COLOR


def _draw_fragment(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top : top + height, left : left + width] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    areas = rng.sample(sorted(AREA_TO_DIMS), pair_count)
    dims = [rng.choice(AREA_TO_DIMS[area]) for area in areas]
    colors = rng.sample(COLOR_POOL, pair_count)

    max_box_h = max(max(inner_h + 2, inner_h) for inner_h, _ in dims)
    max_box_w = max(max(inner_w + 2, inner_w) for _, inner_w in dims)
    min_side = max(18, max_box_h + 4, max_box_w + 4)
    if min_side > 30:
        return None

    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = list(range(pair_count))
    rng.shuffle(order)

    frame_specs: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * pair_count
    frag_specs: list[tuple[int, int, int, int, int]] = [(-1, -1, -1, -1, -1)] * pair_count

    for idx in order:
        inner_h, inner_w = dims[idx]
        frame_top_left = _place_box(rng, reserved, inner_h + 2, inner_w + 2, PLACEMENT_MARGIN)
        if frame_top_left is None:
            return None
        frag_top_left = _place_box(rng, reserved, inner_h, inner_w, PLACEMENT_MARGIN)
        if frag_top_left is None:
            return None

        fr, fc = frame_top_left
        gr, gc = frag_top_left
        frame_specs[idx] = (fr, fc, inner_h, inner_w)
        frag_specs[idx] = (gr, gc, inner_h, inner_w, colors[idx])

    for fr, fc, inner_h, inner_w in frame_specs:
        _draw_frame(grid, fr, fc, inner_h, inner_w)

    for gr, gc, inner_h, inner_w, color in frag_specs:
        _draw_fragment(grid, gr, gc, inner_h, inner_w, color)

    if int(np.count_nonzero(grid == BG)) <= int(np.count_nonzero(grid != BG)):
        return None

    frame_areas = Counter((inner_h * inner_w) for _, _, inner_h, inner_w in frame_specs)
    if any(v != 1 for v in frame_areas.values()):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
