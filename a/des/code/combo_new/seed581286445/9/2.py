from __future__ import annotations

import random

import numpy as np


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    """Return 4-connected monochromatic non-background components."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rectangular_frame(grid: np.ndarray, component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    color = int(component["color"])
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    border = _border_mask(height, width)
    expected_area = int(border.sum())
    if int(component["area"]) != expected_area:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch[border] == color):
        return False
    if np.any(patch[~border] == color):
        return False
    return True


def _contains(outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    components = _find_components(grid, bg=0)
    assert components, "expected at least one non-background component"

    frames: list[dict] = []
    for component in components:
        assert _is_rectangular_frame(grid, component), "every non-black object must be a hollow rectangular frame"
        r0, c0, r1, c1 = component["bbox"]
        frames.append(
            {
                "color": int(component["color"]),
                "bbox": tuple(component["bbox"]),
                "size": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )

    assert len(frames) >= 2, "expected multiple nested frames"
    colors = [frame["color"] for frame in frames]
    assert len(colors) == len(set(colors)), "frame colors must be unique"

    frames.sort(key=lambda frame: (-frame["size"], frame["bbox"]))
    for outer, inner in zip(frames, frames[1:]):
        assert _contains(outer["bbox"], inner["bbox"]), "frames must form one strict nesting chain"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the color order of a chain of nested hollow rectangular frames.

    Input structure:
    - The background is black.
    - The foreground consists of multiple one-cell-thick hollow rectangular frames.
    - The frames have distinct colors and are strictly nested, with black gaps between neighboring frames and inside the innermost frame.

    Rule:
    - Detect every rectangular frame and sort the frames by containment depth from outermost to innermost.
    - Keep every frame's geometry fixed in place.
    - Reassign colors in reverse depth order: the outermost frame gets the original innermost color, the next frame gets the next deepest color, and so on.

    Key insight:
    - The meaningful structure is the containment hierarchy of the frames, not their absolute positions or the numeric ids of their colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    reversed_colors = [frame["color"] for frame in reversed(frames)]

    out = g.copy()
    for frame, new_color in zip(frames, reversed_colors):
        _paint_frame(out, frame["bbox"], new_color)
    return out


# --------------- generator ---------------

FRAME_COUNT_RANGE = (2, 6)
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_SIDE = 30
OUTER_SIDE_MAX = 27
MAX_TOTAL_MARGIN = 6
MAX_GENERATION_ATTEMPTS = 300


def _min_side_for_frame_count(count: int) -> int:
    return 3 + 4 * (count - 1)


def _sample_insets(current_len: int, min_inner_len: int) -> tuple[int, int]:
    slack = int(current_len - min_inner_len - 4)
    assert slack >= 0
    extra_a = random.randint(0, slack)
    extra_b = random.randint(0, slack - extra_a)
    return 2 + extra_a, 2 + extra_b


def _build_nested_bboxes(count: int, outer_h: int, outer_w: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    r0, c0, r1, c1 = 0, 0, outer_h - 1, outer_w - 1

    for depth in range(count):
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
        remaining = count - depth - 1
        if remaining == 0:
            break

        min_inner_h = _min_side_for_frame_count(remaining)
        min_inner_w = _min_side_for_frame_count(remaining)
        top_gap, bottom_gap = _sample_insets(r1 - r0 + 1, min_inner_h)
        left_gap, right_gap = _sample_insets(c1 - c0 + 1, min_inner_w)

        r0 += top_gap
        r1 -= bottom_gap
        c0 += left_gap
        c1 -= right_gap

    return boxes


def _split_margin(cap: int) -> tuple[int, int]:
    before = random.randint(0, cap)
    after = cap - before
    return before, after


def _render_frames(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for bbox, color in zip(boxes, colors):
        _paint_frame(grid, bbox, color)
    return grid


def _attempt_generate() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    min_side = _min_side_for_frame_count(frame_count)
    if min_side > OUTER_SIDE_MAX:
        return None

    outer_h = random.randint(min_side, OUTER_SIDE_MAX)
    outer_w = random.randint(min_side, OUTER_SIDE_MAX)
    local_boxes = _build_nested_bboxes(frame_count, outer_h, outer_w)

    extra_h = random.randint(0, min(MAX_TOTAL_MARGIN, MAX_SIDE - outer_h))
    extra_w = random.randint(0, min(MAX_TOTAL_MARGIN, MAX_SIDE - outer_w))
    top_margin, bottom_margin = _split_margin(extra_h)
    left_margin, right_margin = _split_margin(extra_w)

    placed_boxes = [
        (r0 + top_margin, c0 + left_margin, r1 + top_margin, c1 + left_margin)
        for r0, c0, r1, c1 in local_boxes
    ]
    height = outer_h + top_margin + bottom_margin
    width = outer_w + left_margin + right_margin
    assert 1 <= height <= MAX_SIDE and 1 <= width <= MAX_SIDE

    colors = random.sample(PALETTE, frame_count)
    inp = _render_frames((height, width), placed_boxes, colors)
    expected = _render_frames((height, width), placed_boxes, list(reversed(colors)))

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
