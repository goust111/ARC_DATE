from __future__ import annotations

import numpy as np


BBox = tuple[int, int, int, int]

BG = 0
FRAME_COLORS = (1, 2, 3, 4)
PATTERN_COLORS = (5, 6, 7, 8, 9)
GEN_MAX_ATTEMPTS = 300
MAX_OUTER_DIM = 24
CASE_SCHEDULE = (
    (2, 2),
    (3, 2),
    (4, 2),
    (2, 3),
    (3, 3),
    (4, 3),
)
GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_height(box: BBox) -> int:
    return box[2] - box[0] + 1


def _bbox_width(box: BBox) -> int:
    return box[3] - box[1] + 1


def _bbox_area(box: BBox) -> int:
    return _bbox_height(box) * _bbox_width(box)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.asarray(coords, dtype=int)
            comps.append({"color": color, "coords": coords_arr, "bbox": _bbox(coords_arr)})
    return comps


def _is_hollow_rectangle_component(component: dict) -> bool:
    coords = component["coords"]
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((height, width), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return bool(np.array_equal(mask, _border_mask(height, width)))


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _get_bg(grid)
    frames = []
    for component in _monochromatic_components(grid, bg):
        if _is_hollow_rectangle_component(component):
            r0, c0, r1, c1 = component["bbox"]
            frames.append(
                {
                    "color": int(component["color"]),
                    "bbox": (r0, c0, r1, c1),
                    "coords": component["coords"],
                }
            )
    return bg, frames


def _contains(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _upscale(pattern: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(pattern, factor, axis=0), factor, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the nested hollow rectangles, crop the innermost interior pattern, and enlarge it by the nesting depth.

    Input structure:
    - The background is black.
    - One foreground structure is a set of 2 to 4 single-pixel-thick hollow rectangular frames, each a 4-connected monochrome border.
    - The frames are strictly nested with at least one black gap between consecutive layers, and the innermost frame encloses a small non-black pattern.

    Rule:
    - Identify every monochrome component whose pixels exactly match the border of its bounding box.
    - Keep the frame components and order them by containment so they form one nested chain.
    - Let D be the total number of nested frames, crop the strict interior of the innermost frame, and pixel-upscale that cropped pattern by a factor of D in both axes.

    Key insight:
    - The hollow rectangles are not the output content; their nesting depth encodes the scale factor, while the actual payload is the pattern inside the deepest cavity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, frames = _extract_frames(g)
    assert bg == BG, "expected black background"
    assert 2 <= len(frames) <= 4, "expected 2 to 4 nested frames"

    ordered = sorted(frames, key=lambda frame: _bbox_area(frame["bbox"]))
    for inner, outer in zip(ordered, ordered[1:]):
        assert _contains(outer, inner), "frames must form one strict containment chain"

    depth = len(ordered)
    innermost = ordered[0]["bbox"]
    r0, c0, r1, c1 = innermost
    pattern = g[r0 + 1 : r1, c0 + 1 : c1]
    assert pattern.size > 0, "innermost frame must have a non-empty interior"
    assert np.all(pattern != bg), "innermost interior must be fully occupied by the payload pattern"
    return _upscale(pattern, depth)


def _scheduled_case() -> tuple[int, int, int]:
    global GEN_CALL_COUNT
    idx = GEN_CALL_COUNT
    GEN_CALL_COUNT += 1
    depth, pattern_size = CASE_SCHEDULE[idx % len(CASE_SCHEDULE)]
    mode = (idx // len(CASE_SCHEDULE)) % 3
    return depth, pattern_size, mode


def _sample_pattern(rng: np.random.Generator, size: int) -> np.ndarray:
    colors = np.array(PATTERN_COLORS, dtype=int)
    while True:
        if size == 2:
            cells = rng.choice(colors, size=4, replace=False)
        else:
            chosen = rng.choice(colors, size=5, replace=False)
            extras = rng.choice(chosen, size=4, replace=False)
            cells = np.concatenate([chosen, extras])
        rng.shuffle(cells)
        pattern = np.asarray(cells, dtype=int).reshape(size, size)
        if _pattern_ok(pattern):
            return pattern


def _pattern_ok(pattern: np.ndarray) -> bool:
    if np.any(pattern == BG):
        return False
    unique_count = len({int(v) for v in pattern.ravel()})
    if pattern.shape == (2, 2):
        if unique_count != 4:
            return False
    else:
        if unique_count < 5:
            return False
    for row in pattern:
        if len({int(v) for v in row}) < 2:
            return False
    for col in pattern.T:
        if len({int(v) for v in col}) < 2:
            return False
    if np.array_equal(pattern, np.flipud(pattern)) and np.array_equal(pattern, np.fliplr(pattern)):
        return False
    _, pattern_frames = _extract_frames(pattern)
    return len(pattern_frames) == 0


def _sample_expansion(
    rng: np.random.Generator, mode: int, layer_idx: int
) -> tuple[int, int]:
    if mode == 0:
        row_delta = int(rng.integers(2, 4))
        col_delta = int(rng.integers(2, 4))
    elif mode == 1:
        row_delta = int(rng.integers(2, 4))
        col_delta = int(rng.integers(3, 5))
    else:
        row_delta = int(rng.integers(3, 5))
        col_delta = int(rng.integers(2, 4))
    if (layer_idx + int(rng.integers(2))) % 2 == 1:
        row_delta, col_delta = col_delta, row_delta
    return row_delta, col_delta


def _sample_local_bboxes(
    rng: np.random.Generator, depth: int, pattern_size: int, mode: int
) -> list[BBox]:
    inner_side = pattern_size + 2
    for _ in range(100):
        bboxes = [(0, 0, inner_side - 1, inner_side - 1)]
        for layer_idx in range(1, depth):
            row_delta, col_delta = _sample_expansion(rng, mode, layer_idx)
            pr0, pc0, pr1, pc1 = bboxes[-1]
            bboxes.append(
                (
                    pr0 - row_delta,
                    pc0 - col_delta,
                    pr1 + row_delta,
                    pc1 + col_delta,
                )
            )
        min_r = min(box[0] for box in bboxes)
        min_c = min(box[1] for box in bboxes)
        shifted = [
            (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c)
            for r0, c0, r1, c1 in bboxes
        ]
        outer = shifted[-1]
        if _bbox_height(outer) <= MAX_OUTER_DIM and _bbox_width(outer) <= MAX_OUTER_DIM:
            return shifted
    raise AssertionError("failed to sample nested frame geometry within bounds")


def _offset_bbox(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _check_frame_pixels(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(sub.shape[0], sub.shape[1])
    assert np.all(sub[border] == color), "frame border pixels must match the planned color"


def _validate_instance(
    grid: np.ndarray,
    frame_boxes_outer_to_inner: list[BBox],
    frame_colors_outer_to_inner: list[int],
    pattern: np.ndarray,
) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    outer = frame_boxes_outer_to_inner[0]
    outside = np.ones(grid.shape, dtype=bool)
    outside[outer[0] : outer[2] + 1, outer[1] : outer[3] + 1] = False
    assert np.all(grid[outside] == BG), "no foreground pixels should exist outside the outer frame"

    for box, color in zip(frame_boxes_outer_to_inner, frame_colors_outer_to_inner):
        _check_frame_pixels(grid, box, color)

    for outer_box, inner_box in zip(
        frame_boxes_outer_to_inner, frame_boxes_outer_to_inner[1:]
    ):
        inter = grid[outer_box[0] + 1 : outer_box[2], outer_box[1] + 1 : outer_box[3]]
        clear = np.ones(inter.shape, dtype=bool)
        rr0 = inner_box[0] - (outer_box[0] + 1)
        rr1 = inner_box[2] - (outer_box[0] + 1)
        cc0 = inner_box[1] - (outer_box[1] + 1)
        cc1 = inner_box[3] - (outer_box[1] + 1)
        clear[rr0 : rr1 + 1, cc0 : cc1 + 1] = False
        assert np.all(inter[clear] == BG), "between consecutive frames there must be only background"

    inner = frame_boxes_outer_to_inner[-1]
    payload = grid[inner[0] + 1 : inner[2], inner[1] + 1 : inner[3]]
    assert np.array_equal(payload, pattern), "payload pattern must occupy the full innermost interior"
    assert np.all(payload != BG), "payload pattern must contain no background holes"

    detected_bg, frames = _extract_frames(grid)
    assert detected_bg == BG
    assert len(frames) == len(frame_boxes_outer_to_inner), "independent frame scan must recover every frame"

    expected = _upscale(pattern, len(frame_boxes_outer_to_inner))
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the planned upscale"


def _gen_attempt(depth: int, pattern_size: int, mode: int) -> np.ndarray:
    rng = np.random.default_rng()
    pattern = _sample_pattern(rng, pattern_size)
    local_inner_to_outer = _sample_local_bboxes(rng, depth, pattern_size, mode)
    local_outer_to_inner = list(reversed(local_inner_to_outer))
    outer_local = local_outer_to_inner[0]
    outer_h = _bbox_height(outer_local)
    outer_w = _bbox_width(outer_local)

    min_h = max(outer_h + 1, 18)
    min_w = max(outer_w + 1, 18)
    height = int(rng.integers(min_h, 31))
    width = int(rng.integers(min_w, 31))
    top = int(rng.integers(0, height - outer_h + 1))
    left = int(rng.integers(0, width - outer_w + 1))

    grid = np.full((height, width), BG, dtype=int)
    frame_colors = rng.choice(np.array(FRAME_COLORS, dtype=int), size=depth, replace=False).tolist()
    placed_outer_to_inner = []
    for box, color in zip(local_outer_to_inner, frame_colors):
        placed = _offset_bbox(box, top, left)
        _draw_frame(grid, placed, int(color))
        placed_outer_to_inner.append(placed)

    inner = placed_outer_to_inner[-1]
    grid[inner[0] + 1 : inner[2], inner[1] + 1 : inner[3]] = pattern

    _validate_instance(grid, placed_outer_to_inner, [int(c) for c in frame_colors], pattern)
    return grid


def generate_input() -> np.ndarray:
    depth, pattern_size, mode = _scheduled_case()
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            return _gen_attempt(depth, pattern_size, mode)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid nested-frame instance")
