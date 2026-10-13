import random
from typing import List, Sequence, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]

BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GRID_SIDE_RANGE = (13, 22)
FRAME_COUNT_CHOICES = (3, 4)
MIN_FRAME_SIDE = 5
MAX_GENERATION_ATTEMPTS = 400


def _frame_mask(shape: Tuple[int, int], bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    mask = _frame_mask(grid.shape, bbox)
    grid[mask] = int(color)


def _bbox_of_mask(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_single_pixel_rectangular_frame(mask: np.ndarray) -> bool:
    bbox = _bbox_of_mask(mask)
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if min(height, width) < 3:
        return False
    expected = _frame_mask(mask.shape, bbox)
    return bool(np.array_equal(mask, expected))


def _detect_frames(grid: np.ndarray) -> List[dict]:
    work = np.asarray(grid, dtype=int).copy()
    guide_color = int(work[0, 0])
    work[0, 0] = BACKGROUND

    colors = [int(color) for color in np.unique(work) if int(color) != BACKGROUND]
    if not 3 <= len(colors) <= 4:
        raise AssertionError("expected 3-4 frame colors after removing the guide pixel")

    frames = []
    for color in colors:
        mask = work == color
        if not _is_single_pixel_rectangular_frame(mask):
            raise AssertionError(f"color {color} is not a single-pixel rectangular frame")
        bbox = _bbox_of_mask(mask)
        frames.append(
            {
                "color": color,
                "bbox": bbox,
                "area": int(mask.sum()),
            }
        )

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    for depth, frame in enumerate(frames):
        frame["depth"] = depth

    for outer, inner in zip(frames, frames[1:]):
        r0, c0, r1, c1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        if (ir0, ic0, ir1, ic1) != (r0 + 2, c0 + 2, r1 - 2, c1 - 2):
            raise AssertionError("frames must be nested with exactly one blank cell gap")

    if guide_color not in {frame["color"] for frame in frames}:
        raise AssertionError("guide color must match exactly one frame color")

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular frames by decoding a cyclic shift amount from the top-left guide pixel.

    Input structure:
    - The grid has a black background and contains 3-4 single-pixel-wide nested rectangular frames.
    - Each frame is a different non-black color, and consecutive frames are separated by exactly one black cell gap.
    - The top-left cell (0,0) is a single guide pixel whose color matches one of the frame colors.

    Rule:
    - Detect the nested frames and sort them by depth from outermost to innermost.
    - Read the guide color at (0,0) and find the frame with that color; its depth is the cyclic shift amount.
    - Recolor every frame so depth i receives the original color of depth (i - shift) mod N, where N is the frame count.
    - Erase the guide pixel by setting (0,0) to black.

    Key insight:
    - The corner guide does not name a target color directly; it names a frame, and that frame depth becomes the rotation offset for the full nested sequence.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    frames = _detect_frames(g)
    guide_color = int(g[0, 0])
    shift = next(frame["depth"] for frame in frames if frame["color"] == guide_color)
    colors = [int(frame["color"]) for frame in frames]

    out = g.copy()
    out[0, 0] = BACKGROUND
    for frame in frames:
        depth = int(frame["depth"])
        new_color = colors[(depth - shift) % len(colors)]
        _draw_frame(out, frame["bbox"], new_color)
    return out


def _choose_outer_bbox(height: int, width: int, frame_count: int) -> BBox:
    min_outer_side = MIN_FRAME_SIDE + 4 * (frame_count - 1)
    outer_h = random.randint(min_outer_side, height - 2)
    outer_w = random.randint(min_outer_side, width - 2)
    r0 = random.randint(1, height - outer_h - 1)
    c0 = random.randint(1, width - outer_w - 1)
    return (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)


def _nested_bboxes(outer_bbox: BBox, frame_count: int) -> List[BBox]:
    r0, c0, r1, c1 = outer_bbox
    boxes = []
    for depth in range(frame_count):
        inset = 2 * depth
        boxes.append((r0 + inset, c0 + inset, r1 - inset, c1 - inset))
    return boxes


def _expected_from_construction(
    grid: np.ndarray,
    bboxes: Sequence[BBox],
    colors: Sequence[int],
    guide_depth: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[0, 0] = BACKGROUND
    for depth, bbox in enumerate(bboxes):
        new_color = int(colors[(depth - guide_depth) % len(colors)])
        _draw_frame(out, bbox, new_color)
    return out


def _semantic_checks(
    inp: np.ndarray,
    bboxes: Sequence[BBox],
    colors: Sequence[int],
    guide_depth: int,
) -> None:
    if inp.shape[0] > 30 or inp.shape[1] > 30:
        raise AssertionError("grid exceeds ARC size bound")
    if int(inp[0, 0]) != int(colors[guide_depth]):
        raise AssertionError("guide pixel must copy the chosen frame color")
    if len(set(map(int, colors))) != len(colors):
        raise AssertionError("frame colors must be distinct")

    for outer, inner in zip(bboxes, bboxes[1:]):
        if inner != (outer[0] + 2, outer[1] + 2, outer[2] - 2, outer[3] - 2):
            raise AssertionError("constructed boxes do not have the required 1-cell gaps")

    expected = _expected_from_construction(inp, bboxes, colors, guide_depth)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        raise AssertionError("solver output does not match the construction semantics")

    detected = _detect_frames(inp)
    detected_boxes = [frame["bbox"] for frame in detected]
    detected_colors = [int(frame["color"]) for frame in detected]
    if list(bboxes) != detected_boxes:
        raise AssertionError("detector failed to recover the constructed nesting")
    if list(colors) != detected_colors:
        raise AssertionError("detector failed to recover frame colors in depth order")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*GRID_SIDE_RANGE)
        width = random.randint(*GRID_SIDE_RANGE)
        frame_count = random.choice(FRAME_COUNT_CHOICES)
        min_outer_side = MIN_FRAME_SIDE + 4 * (frame_count - 1)
        if min(height, width) < min_outer_side + 2:
            continue

        outer_bbox = _choose_outer_bbox(height, width, frame_count)
        bboxes = _nested_bboxes(outer_bbox, frame_count)
        colors = random.sample(PALETTE, frame_count)
        guide_depth = random.randrange(frame_count)

        grid = np.zeros((height, width), dtype=int)
        for bbox, color in zip(bboxes, colors):
            _draw_frame(grid, bbox, color)
        grid[0, 0] = int(colors[guide_depth])

        _semantic_checks(grid, bboxes, colors, guide_depth)
        return grid

    raise RuntimeError("failed to generate a valid nested-frame input")
