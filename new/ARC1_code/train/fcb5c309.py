import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste, random_free_location
import random


def _is_rectangular_frame(obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    mask = np.zeros((height, width), dtype=bool)
    coords = obj["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _bbox_area(obj: dict) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the largest rectangular foreground frame and recolor its cropped contents to the marker color.

    Input structure
    - The background is the dominant color.
    - There are exactly two foreground colors: one forms complete rectangular
      one-pixel-thick frames, and the other appears as marker pixels/noise.
    - The intended frame is the unique largest rectangular frame by bounding-box area.

    Rule
    - Find the foreground color that forms rectangular frames.
    - Crop the bounding box of the unique largest frame of that color.
    - In the crop, replace every non-background pixel with the other foreground color.

    Key insight
    - The marker color supplies the output color; only the largest frame's bbox
      decides which marker pixels are kept and which outside pixels are discarded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    frame_candidates_by_color: dict[int, list[dict]] = {}
    for color in colors:
        objects = find_components(g, bg=bg, colors=[color], connectivity=4, monochromatic=True)
        frames = [obj for obj in objects if _is_rectangular_frame(obj)]
        frame_candidates_by_color[int(color)] = frames

    frame_colors = [color for color, frames in frame_candidates_by_color.items() if frames]
    assert len(frame_colors) == 1, "expected frames in exactly one foreground color"

    frame_color = frame_colors[0]
    marker_color = next(color for color in colors if color != frame_color)
    frames = sorted(frame_candidates_by_color[frame_color], key=_bbox_area, reverse=True)
    assert frames, "expected at least one rectangular frame"
    assert len(frames) == 1 or _bbox_area(frames[0]) > _bbox_area(frames[1]), "expected a unique largest frame"

    r0, c0, r1, c1 = frames[0]["bbox"]
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.any(crop == marker_color), "largest frame crop should contain marker pixels"

    return np.where(crop != bg, marker_color, bg).astype(int)

# --------------- generator (o5x) ---------------

# Canvas and palette ranges.
BG = 0
CANVAS_SIZE_RANGE, FOREGROUND_COLORS = (13, 17), tuple(range(1, 10))
# Frame sizing and placement constraints.
LARGE_FRAME_HEIGHT_RANGE, LARGE_FRAME_WIDTH_RANGE = (6, 8), (7, 8)
SMALL_FRAME_COUNT_RANGE = (1, 2)
SMALL_FRAME_SIZE_RANGE = (3, 5)
FRAME_PADDING, FRAME_PADDING_CONNECTIVITY = 1, 4
# Marker counts and component-shape constraints.
LARGE_INTERIOR_MARKER_COUNT_RANGE, OUTSIDE_MARKER_COUNT_RANGE = (2, 4), (7, 17)
SMALL_INTERIOR_MARKER_MAX, OUTSIDE_TWO_CELL_COMPONENT_MAX = 2, 5
MARKER_COMPONENT_COUNT_RANGE, GENERATION_RETRY_LIMIT = (9, 19), 300


def _gen_frame_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _gen_marker_blocked_mask(grid: np.ndarray, marker_color: int) -> np.ndarray:
    marker = grid == marker_color
    blocked = np.zeros(grid.shape, dtype=bool)
    blocked[1:, :] |= marker[:-1, :]
    blocked[:-1, :] |= marker[1:, :]
    blocked[:, 1:] |= marker[:, :-1]
    blocked[:, :-1] |= marker[:, 1:]
    return blocked


def _gen_place_marker_component(
    grid: np.ndarray,
    allowed: np.ndarray,
    marker_color: int,
    size: int,
) -> list[tuple[int, int]] | None:
    H, W = grid.shape
    free = allowed & (grid == BG) & ~_gen_marker_blocked_mask(grid, marker_color)
    if size == 1:
        candidates = [[(int(r), int(c))] for r, c in np.argwhere(free)]
    elif size == 2:
        vertical = free[:-1, :] & free[1:, :]
        horizontal = free[:, :-1] & free[:, 1:]
        candidates = []
        for r in range(H):
            for c in range(W):
                if r + 1 < H and vertical[r, c]:
                    candidates.append([(r, c), (r + 1, c)])
                if c + 1 < W and horizontal[r, c]:
                    candidates.append([(r, c), (r, c + 1)])
    else:
        raise ValueError("unsupported marker component size")
    if not candidates:
        return None
    cells = random.choice(candidates)
    for r, c in cells:
        grid[r, c] = marker_color
    return cells


def _gen_inside_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _gen_place_frames(grid: np.ndarray, reserve: np.ndarray, frame_color: int):
    large_h, large_w = random.randint(*LARGE_FRAME_HEIGHT_RANGE), random.randint(*LARGE_FRAME_WIDTH_RANGE)
    small_count = random.randint(*SMALL_FRAME_COUNT_RANGE)
    small_sizes = [(random.randint(*SMALL_FRAME_SIZE_RANGE), random.randint(*SMALL_FRAME_SIZE_RANGE)) for _ in range(small_count)]
    frame_specs = [(large_h, large_w, True), *[(h, w, False) for h, w in small_sizes]]
    boxes: list[tuple[int, int, int, int, bool]] = []
    for height, width, is_large in frame_specs:
        full_box = np.ones((height, width), dtype=int)
        try:
            r0, c0 = random_free_location(
                reserve,
                full_box,
                transparent=BG,
                padding=FRAME_PADDING,
                padding_connectivity=FRAME_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None, None
        paste(reserve, full_box, r0, c0)
        paste(grid, _gen_frame_patch(height, width, frame_color), r0, c0, transparent=BG)
        boxes.append((r0, c0, r0 + height - 1, c0 + width - 1, is_large))
    return boxes, (large_h, large_w)


def _gen_add_largest_markers(grid: np.ndarray, output: np.ndarray, large_box, marker_color: int) -> int | None:
    lr0, lc0, lr1, lc1, _ = large_box
    marker_count = random.randint(*LARGE_INTERIOR_MARKER_COUNT_RANGE)
    allowed = _gen_inside_mask(grid.shape, (lr0, lc0, lr1, lc1))
    for _ in range(marker_count):
        cells = _gen_place_marker_component(grid, allowed, marker_color, 1)
        if cells is None:
            return None
        r, c = cells[0]
        output[r - lr0, c - lc0] = marker_color
    return marker_count


def _gen_add_small_frame_markers(grid: np.ndarray, boxes, marker_color: int) -> int | None:
    small_boxes = [(r0, c0, r1, c1) for r0, c0, r1, c1, is_large in boxes if not is_large]
    marker_count = random.randint(0, min(SMALL_INTERIOR_MARKER_MAX, len(small_boxes)))
    random.shuffle(small_boxes)
    for box in small_boxes[:marker_count]:
        small_allowed = _gen_inside_mask(grid.shape, box)
        if _gen_place_marker_component(grid, small_allowed, marker_color, 1) is None:
            return None
    return marker_count


def _gen_add_outside_markers(grid: np.ndarray, reserve: np.ndarray, marker_color: int, interior_marker_count: int) -> bool:
    outside_total = random.randint(*OUTSIDE_MARKER_COUNT_RANGE)
    min_components, max_components = MARKER_COMPONENT_COUNT_RANGE
    min_twos = max(0, interior_marker_count + outside_total - max_components)
    max_twos = min(
        OUTSIDE_TWO_CELL_COMPONENT_MAX,
        outside_total // 2,
        interior_marker_count + outside_total - min_components,
    )
    if min_twos > max_twos:
        return False
    two_components = random.randint(min_twos, max_twos)
    outside_component_sizes = [2] * two_components + [1] * (outside_total - 2 * two_components)
    random.shuffle(outside_component_sizes)
    outside_allowed = reserve == BG
    for size in outside_component_sizes:
        if _gen_place_marker_component(grid, outside_allowed, marker_color, size) is None:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H, W = random.randint(*CANVAS_SIZE_RANGE), random.randint(*CANVAS_SIZE_RANGE)
    frame_color, marker_color = random.sample(FOREGROUND_COLORS, 2)
    grid = np.full((H, W), BG, dtype=int)
    reserve = np.full((H, W), BG, dtype=int)

    boxes, large_shape = _gen_place_frames(grid, reserve, frame_color)
    if boxes is None:
        return None

    large_box = next(box for box in boxes if box[4])
    output = _gen_frame_patch(*large_shape, marker_color)
    large_marker_count = _gen_add_largest_markers(grid, output, large_box, marker_color)
    if large_marker_count is None:
        return None

    small_marker_count = _gen_add_small_frame_markers(grid, boxes, marker_color)
    if small_marker_count is None:
        return None

    if not _gen_add_outside_markers(grid, reserve, marker_color, large_marker_count + small_marker_count):
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fcb5c309 input")
