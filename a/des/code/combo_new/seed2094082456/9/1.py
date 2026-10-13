import random

import numpy as np


SIDES = ("top", "left", "bottom", "right")
FRAME_COUNT_CHOICES = (3, 4)
INNER_SIDE_RANGE = (3, 5)
PADDING_RANGE = (1, 8)
GENERATION_ATTEMPTS = 500


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    h, w = _bbox_size(bbox)
    return int(h * w)


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer != inner
        and outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _shared_side(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> str | None:
    sides = []
    if inner[0] == outer[0]:
        sides.append("top")
    if inner[1] == outer[1]:
        sides.append("left")
    if inner[2] == outer[2]:
        sides.append("bottom")
    if inner[3] == outer[3]:
        sides.append("right")
    return sides[0] if len(sides) == 1 else None


def _shared_segment(body: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return body[r0, c0 : c1 + 1]
    if side == "left":
        return body[r0 : r1 + 1, c0]
    if side == "bottom":
        return body[r1, c0 : c1 + 1]
    return body[r0 : r1 + 1, c1]


def _pack_frames(bg: int, items: list[tuple[int, tuple[int, int, int, int]]]) -> np.ndarray:
    heights = [_bbox_size(bbox)[0] for _color, bbox in items]
    widths = [_bbox_size(bbox)[1] for _color, bbox in items]
    out_h = max(heights)
    out_w = sum(widths) + max(0, len(items) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    cursor = 0
    for color, bbox in items:
        h, w = _bbox_size(bbox)
        _draw_frame(out, (0, cursor, h - 1, cursor + w - 1), color)
        cursor += w + 1

    return out


def _topological_order(
    frames_by_color: dict[int, dict[str, tuple[int, int, int, int] | int]],
    edges: dict[int, set[int]],
) -> list[int]:
    colors = list(frames_by_color)
    indegree = {color: 0 for color in colors}
    for src, dsts in edges.items():
        for dst in dsts:
            indegree[dst] += 1

    ready = [color for color in colors if indegree[color] == 0]
    ready.sort(key=lambda color: _bbox_area(frames_by_color[color]["bbox"]))  # type: ignore[index]

    order: list[int] = []
    while ready:
        color = ready.pop(0)
        order.append(color)
        for dst in sorted(edges[color], key=lambda c: _bbox_area(frames_by_color[c]["bbox"])):  # type: ignore[index]
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
                ready.sort(key=lambda c: _bbox_area(frames_by_color[c]["bbox"]))  # type: ignore[index]

    if len(order) != len(colors):
        return sorted(colors, key=lambda color: _bbox_area(frames_by_color[color]["bbox"]))  # type: ignore[index]
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the marked occluding nested frames and lay the selected ones out from inner to outer.

    Input structure:
    - A background grid contains several one-cell-thick rectangular frames in different colors.
    - The frames are strictly nested, and each adjacent pair shares exactly one border line
      where the inner frame overwrites the outer one.
    - The last row is a marker strip made of isolated colored cells; each marker color matches
      one frame color.

    Rule:
    - Ignore the marker strip while locating the frames, and recover each frame's full bbox from
      the visible pixels of its color.
    - Use the overwritten shared borders to infer the nesting chain: on a shared side, the color
      visible across the whole inner span is the deeper inner frame.
    - Keep only the frame colors that appear in the marker strip, restore each kept frame as a
      complete rectangle outline, and pack them left-to-right in inner-to-outer order with one
      background column between neighboring frames.

    Key insight:
    - The overlap is informative because the inner frame is the one that remains visible on the
      shared side, so the occlusion graph reveals depth before the marker strip filters the result.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 3

    bg = int(g[0, 0])
    body = g[:-1]
    marker_row = g[-1]

    frame_colors = [int(color) for color in np.unique(body) if int(color) != bg]
    assert frame_colors, "expected at least one frame color above the marker strip"

    frames_by_color: dict[int, dict[str, tuple[int, int, int, int] | int]] = {}
    for color in frame_colors:
        coords = np.argwhere(body == int(color))
        assert coords.size > 0
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each frame must be at least 3x3"
        for r, c in coords:
            assert r in (r0, r1) or c in (c0, c1), "visible frame pixels must stay on the perimeter"
        frames_by_color[int(color)] = {"color": int(color), "bbox": bbox}

    edges = {color: set() for color in frame_colors}
    for i, color_a in enumerate(frame_colors):
        bbox_a = frames_by_color[color_a]["bbox"]  # type: ignore[index]
        for color_b in frame_colors[i + 1 :]:
            bbox_b = frames_by_color[color_b]["bbox"]  # type: ignore[index]
            if _contains_bbox(bbox_a, bbox_b):  # type: ignore[arg-type]
                outer_color, outer_bbox = color_a, bbox_a
                inner_color, inner_bbox = color_b, bbox_b
            elif _contains_bbox(bbox_b, bbox_a):  # type: ignore[arg-type]
                outer_color, outer_bbox = color_b, bbox_b
                inner_color, inner_bbox = color_a, bbox_a
            else:
                continue

            side = _shared_side(inner_bbox, outer_bbox)  # type: ignore[arg-type]
            if side is None:
                continue
            segment = _shared_segment(body, inner_bbox, side)  # type: ignore[arg-type]
            if bool(np.all(segment == int(inner_color))):
                edges[int(inner_color)].add(int(outer_color))

    order = _topological_order(frames_by_color, edges)

    marker_colors = {int(color) for color in marker_row if int(color) != bg}
    assert marker_colors, "marker strip must select at least one frame"
    selected = [
        (color, frames_by_color[color]["bbox"])  # type: ignore[index]
        for color in order
        if color in marker_colors
    ]
    assert selected, "at least one marked frame must be present above the marker strip"

    return _pack_frames(bg, selected)  # type: ignore[arg-type]


def _grow_outer(inner: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = inner
    if side == "top":
        return int(r0), int(c0 - 1), int(r1 + 1), int(c1 + 1)
    if side == "left":
        return int(r0 - 1), int(c0), int(r1 + 1), int(c1 + 1)
    if side == "bottom":
        return int(r0 - 1), int(c0 - 1), int(r1), int(c1 + 1)
    return int(r0 - 1), int(c0 - 1), int(r1 + 1), int(c1)


def _build_chain(n_frames: int) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    inner_h = random.randint(*INNER_SIDE_RANGE)
    inner_w = random.randint(*INNER_SIDE_RANGE)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    sides = random.sample(SIDES, n_frames - 1)
    for side in sides:
        boxes.append(_grow_outer(boxes[-1], side))
    min_r = min(bbox[0] for bbox in boxes)
    min_c = min(bbox[1] for bbox in boxes)
    boxes = [_shift_bbox(bbox, -min_r, -min_c) for bbox in boxes]
    return boxes, sides


def _sample_marker_positions(width: int, count: int) -> list[int]:
    spaced = list(range(1, max(1, width - 1), 2))
    candidates = spaced if len(spaced) >= count else list(range(width))
    return sorted(random.sample(candidates, count))


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    frame_colors = random.sample([color for color in range(10) if color != bg], n_frames)
    boxes, _sides = _build_chain(n_frames)

    keep_count = random.randint(1, n_frames - 1)
    kept_indices = sorted(random.sample(range(n_frames), keep_count))
    expected_items = [(frame_colors[idx], boxes[idx]) for idx in kept_indices]
    expected = _pack_frames(bg, expected_items)
    if max(expected.shape) > 30:
        return None

    outer_h, outer_w = _bbox_size(boxes[-1])
    top = random.randint(*PADDING_RANGE)
    bottom = random.randint(*PADDING_RANGE)
    left = random.randint(*PADDING_RANGE)
    right = random.randint(*PADDING_RANGE)
    width = outer_w + left + right
    min_marker_width = 2 * keep_count + 1
    if width < min_marker_width:
        width = min_marker_width
        right += min_marker_width - (outer_w + left + right)
    height = outer_h + top + bottom + 1
    if height > 30 or width > 30:
        return None

    shifted_boxes = [_shift_bbox(bbox, top, left) for bbox in boxes]
    grid = np.full((height, width), int(bg), dtype=int)

    for color, bbox in zip(reversed(frame_colors), reversed(shifted_boxes)):
        _draw_frame(grid, bbox, color)

    marker_colors = [frame_colors[idx] for idx in kept_indices]
    random.shuffle(marker_colors)
    marker_positions = _sample_marker_positions(width, keep_count)
    for color, col in zip(marker_colors, marker_positions):
        grid[-1, col] = int(color)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame occlusion input")
