from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
LINE_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    groups: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            groups.append(np.array(coords, dtype=int))
    return groups


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    return _bbox_h(box) * _bbox_w(box)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _frame_components(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color in (BG, LINE_COLOR):
            continue
        for coords in _components(grid == color):
            box = _bbox(coords)
            height = _bbox_h(box)
            width = _bbox_w(box)
            if height < 3 or width < 3:
                continue
            border_area = 2 * (height + width) - 4
            if len(coords) != border_area:
                continue

            r0, c0, r1, c1 = box
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((height, width), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue
            frames.append({"color": color, "bbox": box})
    return frames


def _palette_from_nested_frames(grid: np.ndarray) -> list[int]:
    frames = _frame_components(grid)
    assert 2 <= len(frames) <= 4, "expected 2-4 nested rectangle frames"
    frames.sort(key=lambda item: _bbox_area(item["bbox"]), reverse=True)
    for outer, inner in zip(frames, frames[1:]):
        assert _contains(outer["bbox"], inner["bbox"]), "frame bboxes must form one nested chain"
    return [int(frame["color"]) for frame in frames]


def _line_segments(grid: np.ndarray) -> list[dict]:
    segments: list[dict] = []
    for coords in _components(grid == LINE_COLOR):
        box = _bbox(coords)
        height = _bbox_h(box)
        width = _bbox_w(box)
        if height == 1 and width >= 2:
            assert len(coords) == width, "horizontal line must be contiguous"
            segments.append({"orientation": "h", "bbox": box})
        elif width == 1 and height >= 2:
            assert len(coords) == height, "vertical line must be contiguous"
            segments.append({"orientation": "v", "bbox": box})
        else:
            raise AssertionError("gray components must be one-pixel-thick straight segments")
    assert segments, "expected at least one gray segment"
    return segments


def _paint_segment(out: np.ndarray, segment: dict, palette: list[int]) -> None:
    parts = len(palette)
    r0, c0, r1, c1 = map(int, segment["bbox"])
    if segment["orientation"] == "h":
        length = c1 - c0 + 1
    else:
        length = r1 - r0 + 1

    assert length >= parts, "segment must be at least as long as the palette"
    base = length // parts
    assert base >= 1

    index = 0
    for i, color in enumerate(palette):
        run = base if i < parts - 1 else length - base * (parts - 1)
        for _ in range(run):
            if segment["orientation"] == "h":
                out[r0, c0 + index] = color
            else:
                out[r0 + index, c0] = color
            index += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the colors of nested rectangle frames and use them to recolor gray line segments in order.

    Input structure:
    - The background is black.
    - One side of the grid contains 2-4 one-pixel-thick nested rectangular frames, each with a different border color and black gaps between layers.
    - Elsewhere in the grid, every gray foreground component is an isolated horizontal or vertical one-pixel-thick line segment.

    Rule:
    - Detect the nested frames and read their border colors from the outermost layer to the innermost layer to obtain an ordered palette.
    - For each gray line segment, split its length into as many consecutive parts as there are palette colors; all earlier parts have size floor(length / palette_size), and the last part absorbs any remainder.
    - Recolor each segment from its start end using the palette order: horizontal segments run left-to-right and vertical segments run top-to-bottom.
    - Leave the nested frames and the black background unchanged.

    Key insight:
    - The nested frames are not target objects themselves; they encode a visible color sequence that must be transferred, in order, onto separate gray carriers with orientation-dependent start points.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    palette = _palette_from_nested_frames(g)
    segments = _line_segments(g)

    out = g.copy()
    for segment in segments:
        _paint_segment(out, segment, palette)
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    height, width = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    height, width = occupied.shape
    if r0 < 0 or c0 < 0 or r1 >= height or c1 >= width:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _orientation_plan() -> list[str]:
    mode = random.choice(("horizontal", "vertical", "mixed"))
    if mode == "horizontal":
        return ["h"] * random.randint(2, 4)
    if mode == "vertical":
        return ["v"] * random.randint(2, 3)
    h_count = random.randint(1, 2)
    v_count = random.randint(1, 2)
    plan = ["h"] * h_count + ["v"] * v_count
    random.shuffle(plan)
    return plan


def _divisibility_targets(count: int) -> list[bool]:
    mode = random.choice(("divisible", "nondivisible", "mixed"))
    if mode == "divisible":
        return [True] * count
    if mode == "nondivisible":
        return [False] * count
    targets = [True] * (count // 2) + [False] * (count - count // 2)
    random.shuffle(targets)
    return targets


def _place_segment(
    grid: np.ndarray,
    occupied: np.ndarray,
    orientation: str,
    want_divisible: bool,
    layers: int,
    right_c0: int,
    right_c1: int,
) -> int | None:
    height = grid.shape[0]
    if orientation == "h":
        max_len = right_c1 - right_c0 + 1
    else:
        max_len = height

    lengths = [length for length in range(layers, max_len + 1) if (length % layers == 0) == want_divisible]
    if not lengths:
        return None

    for _ in range(80):
        length = random.choice(lengths)
        if orientation == "h":
            row = random.randrange(height)
            col = random.randrange(right_c0, right_c1 - length + 2)
            if not _can_place(occupied, row, col, row, col + length - 1, pad=1):
                continue
            grid[row, col : col + length] = LINE_COLOR
            _reserve(occupied, row, col, row, col + length - 1, pad=1)
            return length

        col = random.randrange(right_c0, right_c1 + 1)
        row = random.randrange(0, height - length + 1)
        if not _can_place(occupied, row, col, row + length - 1, col, pad=1):
            continue
        grid[row : row + length, col] = LINE_COLOR
        _reserve(occupied, row, col, row + length - 1, col, pad=1)
        return length

    return None


def _gen_attempt() -> np.ndarray | None:
    layers = random.randint(2, 4)
    palette = random.sample(COLOR_POOL, layers)
    plan = _orientation_plan()
    h_count = plan.count("h")
    v_count = plan.count("v")

    outer_min = 4 * layers - 1
    outer_w_max = 15
    outer_h_max = 18
    if outer_w_max < outer_min or outer_h_max < outer_min:
        return None
    outer_w = random.randint(outer_min, outer_w_max)
    outer_h = random.randint(outer_min, outer_h_max)

    left_margin = random.randint(0, 1)
    right_margin = random.randint(0, 1)
    gap = random.randint(3, 5)
    right_w_min = max(layers + 2, 6, 3 * v_count, left_margin + outer_w - gap - right_margin)
    right_w_max = min(14, 30 - left_margin - outer_w - gap - right_margin)
    if right_w_max < right_w_min:
        return None
    right_w = random.randint(right_w_min, right_w_max)
    width = left_margin + outer_w + gap + right_w + right_margin

    min_height = max(outer_h, 3 * h_count - 1, layers + 2)
    max_height = min(30, min_height + 8)
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)

    top = random.randint(0, height - outer_h)
    left = left_margin
    right_c0 = left + outer_w + gap
    right_c1 = width - right_margin - 1

    grid = np.zeros((height, width), dtype=int)
    for i, color in enumerate(palette):
        inset = 2 * i
        box = (
            top + inset,
            left + inset,
            top + outer_h - 1 - inset,
            left + outer_w - 1 - inset,
        )
        _draw_frame(grid, box, int(color))

    occupied = np.zeros((height, width), dtype=bool)
    _reserve(occupied, top, left, top + outer_h - 1, left + outer_w - 1, pad=1)

    targets = _divisibility_targets(len(plan))
    lengths: list[int] = []
    for orientation, want_divisible in zip(plan, targets):
        length = _place_segment(grid, occupied, orientation, want_divisible, layers, right_c0, right_c1)
        if length is None:
            return None
        lengths.append(length)

    if len(set(lengths)) == 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame palette input")
