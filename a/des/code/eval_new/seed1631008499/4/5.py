from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_COUNT_OPTIONS = (4,)
GEN_ATTEMPTS = 300

SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
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


def _frame_info(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None

    rel = coords - np.array([r0, c0], dtype=int)
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    if not bool(np.all(on_border)):
        return None

    color = int(grid[int(coords[0, 0]), int(coords[0, 1])])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not bool(np.all(sub[0, :] == color)):
        return None
    if not bool(np.all(sub[-1, :] == color)):
        return None
    if not bool(np.all(sub[:, 0] == color)):
        return None
    if not bool(np.all(sub[:, -1] == color)):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "pixels": int(coords.shape[0]),
        "bbox_area": h * w,
        "coords": coords,
    }


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    frame_mask = np.zeros_like(grid, dtype=bool)
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, GREY):
            continue
        for comp in _components(grid == color):
            info = _frame_info(grid, comp)
            if info is None:
                continue
            frames.append(info)
            frame_mask[comp[:, 0], comp[:, 1]] = True

    assert len(frames) >= 2, "expected a nested frame chain"
    frames.sort(key=lambda frame: int(frame["bbox_area"]), reverse=True)
    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "frames must be strictly nested"
    return frames, frame_mask


def _legend_colors(grid: np.ndarray, bg: int, frame_mask: np.ndarray, frame_count: int) -> list[int]:
    coords = np.argwhere((grid != bg) & (grid != GREY) & (~frame_mask))
    assert int(coords.shape[0]) == frame_count, "legend must contain one pixel per frame"
    cols = {int(c) for c in coords[:, 1]}
    rows = {int(r) for r in coords[:, 0]}
    assert len(cols) == 1 and len(rows) == frame_count, "legend must be a vertical pixel sequence"
    coords = coords[np.argsort(coords[:, 0])]
    return [int(grid[int(r), int(c)]) for r, c in coords]


def _strictly_inside(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        np.all((coords[:, 0] > r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1))
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey objects from nested-frame depth, using forward or reversed legend order based on the innermost enclosing frame size class.

    Input structure:
    - The background is the most common color, typically black.
    - Several monochrome 1-pixel rectangular frames form one strict nesting chain, and a vertical legend of single colored pixels sits to the right.
    - Grey connected components appear inside the nested structure, either in the innermost chamber or in the bands between consecutive frames.

    Rule:
    - Read the right-side legend from top to bottom as the colors for depths 1, 2, 3, ...
    - For each grey object, count how many frames strictly enclose it; this is its nesting depth, counted from the object's own innermost enclosing frame outward.
    - Rank frame colors by total frame-pixel count and split them into a larger half and a smaller half.
    - If the object's innermost enclosing frame is in the larger half, recolor the object with the legend color at its depth in forward order; otherwise use the legend in reverse order.
    - Keep all frames, the legend, and the background unchanged.

    Key insight:
    - The output color depends on two independent cues: depth comes from the nesting hierarchy, while forward-vs-reverse legend reading is chosen by whether the innermost enclosing frame belongs to the large-perimeter half or the small-perimeter half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    frames, frame_mask = _find_frames(g, bg)
    legend = _legend_colors(g, bg, frame_mask, len(frames))

    ordered_by_size = sorted(frames, key=lambda frame: int(frame["pixels"]), reverse=True)
    large_count = (len(frames) + 1) // 2
    large_colors = {int(frame["color"]) for frame in ordered_by_size[:large_count]}

    out = g.copy()
    grey_objects = _components(g == GREY)
    assert grey_objects, "expected at least one grey object"

    for comp in grey_objects:
        enclosing = [frame for frame in frames if _strictly_inside(comp, frame["bbox"])]
        depth = len(enclosing)
        assert 1 <= depth <= len(frames), "grey object must be inside the nesting chain"
        innermost = min(enclosing, key=lambda frame: int(frame["bbox_area"]))
        if int(innermost["color"]) in large_colors:
            color = int(legend[depth - 1])
        else:
            color = int(legend[-depth])
        out[comp[:, 0], comp[:, 1]] = color

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(mask, dtype=bool)
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
            out |= shifted
    return out


def _interior_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _region_mask(shape: tuple[int, int], frames: list[tuple[int, int, int, int]], depth: int) -> np.ndarray:
    frame_index = depth - 1
    region = _interior_mask(shape, frames[frame_index])
    if frame_index + 1 < len(frames):
        region &= ~_interior_mask(shape, frames[frame_index + 1])
    return region


def _place_shape(
    grid: np.ndarray,
    grey_occ: np.ndarray,
    allowed: np.ndarray,
    shape_mask: np.ndarray,
) -> np.ndarray | None:
    blocked = _dilate8(grey_occ)
    shape_cells = np.argwhere(shape_mask)
    sh, sw = shape_mask.shape
    rows = list(range(grid.shape[0] - sh + 1))
    cols = list(range(grid.shape[1] - sw + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            coords = shape_cells + np.array([r0, c0], dtype=int)
            rr = coords[:, 0]
            cc = coords[:, 1]
            if not bool(np.all(allowed[rr, cc])):
                continue
            if not bool(np.all(grid[rr, cc] == BG)):
                continue
            if bool(np.any(blocked[rr, cc])):
                continue
            grid[rr, cc] = GREY
            grey_occ[rr, cc] = True
            return coords
    return None


def _manual_output(
    grid: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    frame_colors: list[int],
    legend_colors: list[int],
    objects: list[dict],
) -> np.ndarray:
    out = grid.copy()
    sizes = []
    for bbox, color in zip(frames, frame_colors):
        r0, c0, r1, c1 = bbox
        pixel_count = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        sizes.append((pixel_count, int(color)))
    sizes.sort(reverse=True)
    large_count = (len(frames) + 1) // 2
    large_colors = {color for _count, color in sizes[:large_count]}

    for obj in objects:
        depth = int(obj["depth"])
        innermost_color = int(frame_colors[depth - 1])
        color = legend_colors[depth - 1] if innermost_color in large_colors else legend_colors[-depth]
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _sample_geometry(
    frame_count: int,
) -> tuple[int, int, int, int, list[int], list[int], int, int, int, int] | None:
    for _ in range(100):
        inner_h = random.randint(5, 10 if frame_count == 2 else 7)
        inner_w = random.randint(5, 11 if frame_count == 2 else 7)
        row_gaps = [random.randint(3, 4) for _ in range(frame_count - 1)]
        col_gaps = [random.randint(3, 4) for _ in range(frame_count - 1)]
        outer_h = inner_h + 2 * sum(row_gaps)
        outer_w = inner_w + 2 * sum(col_gaps)

        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        legend_gap = random.randint(2, 4)

        total_h = outer_h + top_margin + bottom_margin
        total_w = outer_w + left_margin + right_margin + legend_gap + 1
        if total_h <= 30 and total_w <= 30:
            return (
                total_h,
                total_w,
                outer_h,
                outer_w,
                row_gaps,
                col_gaps,
                top_margin,
                left_margin,
                right_margin,
                legend_gap,
            )
    return None


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    geometry = _sample_geometry(frame_count)
    if geometry is None:
        return None

    total_h, total_w, outer_h, outer_w, row_gaps, col_gaps, top_margin, left_margin, right_margin, legend_gap = geometry
    frame_colors = random.sample(COLOR_POOL, frame_count)
    legend_colors = random.sample([c for c in COLOR_POOL if c not in frame_colors], frame_count)

    grid = np.full((total_h, total_w), BG, dtype=int)
    bboxes: list[tuple[int, int, int, int]] = []
    row_inset = 0
    col_inset = 0
    for level in range(frame_count):
        r0 = top_margin + row_inset
        c0 = left_margin + col_inset
        r1 = top_margin + outer_h - 1 - row_inset
        c1 = left_margin + outer_w - 1 - col_inset
        bboxes.append((r0, c0, r1, c1))
        if level < frame_count - 1:
            row_inset += row_gaps[level]
            col_inset += col_gaps[level]

    for bbox, color in zip(bboxes, frame_colors):
        _draw_frame(grid, bbox, int(color))

    legend_col = int(bboxes[0][3] + legend_gap)
    if legend_col >= total_w - right_margin:
        return None
    legend_row0 = random.randint(0, total_h - frame_count)
    for i, color in enumerate(legend_colors):
        grid[legend_row0 + i, legend_col] = int(color)

    grey_occ = np.zeros_like(grid, dtype=bool)
    objects: list[dict] = []

    for depth in range(1, frame_count + 1):
        allowed = _region_mask(grid.shape, bboxes, depth)
        shapes = list(SHAPES)
        random.shuffle(shapes)
        coords = None
        for shape in shapes:
            coords = _place_shape(grid, grey_occ, allowed, shape)
            if coords is not None:
                break
        if coords is None:
            return None
        objects.append({"depth": depth, "coords": coords})

    extra_objects = random.randint(0, frame_count)
    for _ in range(extra_objects):
        depth = random.randint(1, frame_count)
        allowed = _region_mask(grid.shape, bboxes, depth)
        shapes = list(SHAPES)
        random.shuffle(shapes)
        for shape in shapes:
            coords = _place_shape(grid, grey_occ, allowed, shape)
            if coords is not None:
                objects.append({"depth": depth, "coords": coords})
                break

    expected = _manual_output(grid, bboxes, frame_colors, legend_colors, objects)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
