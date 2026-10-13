from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg


GRID_SIDE_BOUNDS = (16, 29)
SEGMENT_COUNT_BOUNDS = (3, 5)
SEGMENT_LENGTH_MIN = 2
FRAME_INTERIOR_H_BOUNDS = (3, 8)
FRAME_INTERIOR_W_BOUNDS = (4, 10)
OBJECT_EDGE_MARGIN = 1
OBJECT_GAP = 1
GEN_RETRY_LIMIT = 400


def _segment_length(component: dict) -> int:
    r0, c0, r1, c1 = component["bbox"]
    assert int(r0) == int(r1), "segment must be horizontal"
    return int(c1 - c0 + 1)


def _is_hollow_rect_component(grid: np.ndarray, component: dict) -> bool:
    color = int(component["color"])
    area = int(component["area"])
    r0, c0, r1, c1 = component["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False
    if area != 2 * (height + width) - 4:
        return False

    patch = crop_bbox(grid, component["bbox"])
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and np.all(patch[~border] != color))


def _extract_objects(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frame = None
    segments: list[dict] = []
    for component in components:
        if _is_hollow_rect_component(g, component):
            assert frame is None, "expected exactly one hollow rectangle frame"
            frame = component
            continue

        r0, c0, r1, c1 = component["bbox"]
        area = int(component["area"])
        assert int(r0) == int(r1), "every non-frame object must be a horizontal segment"
        assert area == int(c1 - c0 + 1), "segment must be contiguous and one cell thick"
        segments.append(component)

    assert frame is not None, "expected one hollow rectangle frame"
    assert segments, "expected at least one segment"

    frame_color = int(frame["color"])
    segment_colors = [int(component["color"]) for component in segments]
    assert frame_color not in segment_colors, "frame color must be distinct from segment colors"
    assert len(set(segment_colors)) == len(segment_colors), "segment colors must be unique"

    lengths = [_segment_length(component) for component in segments]
    assert len(set(lengths)) == len(lengths), "segment lengths must be unique"

    r0, c0, r1, c1 = frame["bbox"]
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "frame interior must be empty"
    return bg, frame, segments


def _render_cycle_fill(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color_cycle: list[int],
    *,
    anchor_mode: str,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = frame_bbox
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    rr, cc = np.indices(inner.shape)
    period = len(color_cycle)
    palette = np.array(color_cycle, dtype=int)

    if anchor_mode == "local":
        indices = (rr + cc) % period
    elif anchor_mode == "global":
        indices = ((rr + (r0 + 1)) + (cc + (c0 + 1))) % period
    elif anchor_mode == "outer_bbox":
        indices = (rr + cc + 2) % period
    else:
        raise ValueError(f"unknown anchor_mode: {anchor_mode}")

    inner[:, :] = palette[indices]
    return out


def _row_major_cycle(segments: list[dict]) -> list[int]:
    return [int(component["color"]) for component in sorted(segments, key=lambda comp: comp["bbox"][:2])]


def _color_id_cycle(segments: list[dict]) -> list[int]:
    return [int(component["color"]) for component in sorted(segments, key=lambda comp: int(comp["color"]))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the hollow rectangle interior with a diagonal color cycle ordered by segment length.

    Input structure:
    - The grid has one background color, one one-cell-thick hollow rectangle frame, and several scattered one-cell-thick horizontal segments.
    - Each segment has a unique color and a unique length.
    - The frame color is distinct from every segment color and from the background.

    Rule:
    - Sort the segment colors by increasing segment length to obtain a color cycle.
    - Fill each interior cell of the hollow rectangle by the residue of its local diagonal coordinate: the frame interior's top-left cell gets the first cycle color, and each step down or right advances one step in the cycle.
    - Keep the rectangle border, every segment, and the outside background unchanged.

    Key insight:
    - Segment positions are irrelevant except for identifying the horizontal bars; only their lengths define the ordered color cycle, and the diagonal phase is anchored to the rectangle interior rather than to global grid coordinates.
    """
    g = np.asarray(grid, dtype=int)
    _bg, frame, segments = _extract_objects(g)
    cycle = [int(component["color"]) for component in sorted(segments, key=_segment_length)]
    return _render_cycle_fill(g, frame["bbox"], cycle, anchor_mode="local")


def _bbox_is_free(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = blocked.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    h, w = blocked.shape
    rr0 = max(0, r0 - OBJECT_GAP)
    cc0 = max(0, c0 - OBJECT_GAP)
    rr1 = min(h - 1, r1 + OBJECT_GAP)
    cc1 = min(w - 1, c1 + OBJECT_GAP)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_color_cycle(count: int) -> tuple[int, int, list[int]]:
    palette = list(range(10))
    for _ in range(200):
        bg, frame_color, *segment_colors = random.sample(palette, count + 2)
        if segment_colors != sorted(segment_colors) and segment_colors != sorted(segment_colors, reverse=True):
            return int(bg), int(frame_color), [int(color) for color in segment_colors]
    raise RuntimeError("failed to sample a non-monotone color cycle")


def _build_input_grid(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    frame_color: int,
    placed_segments: list[dict],
    bg: int,
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), int(bg), dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    for segment in placed_segments:
        r = int(segment["row"])
        c0 = int(segment["col"])
        c1 = c0 + int(segment["length"])
        grid[r, c0:c1] = int(segment["color"])
    return grid


def _sample_frame_bbox(height: int, width: int, outer_h: int, outer_w: int, period: int) -> tuple[int, int, int, int] | None:
    positions = []
    for r0 in range(OBJECT_EDGE_MARGIN, height - outer_h - OBJECT_EDGE_MARGIN + 1):
        for c0 in range(OBJECT_EDGE_MARGIN, width - outer_w - OBJECT_EDGE_MARGIN + 1):
            if (r0 + c0 + 2) % period == 0:
                continue
            positions.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))
    return random.choice(positions) if positions else None


def _candidate_segment_bboxes(blocked: np.ndarray, length: int) -> list[tuple[int, int, int, int]]:
    height, width = blocked.shape
    boxes = []
    for r in range(OBJECT_EDGE_MARGIN, height - OBJECT_EDGE_MARGIN):
        for c0 in range(OBJECT_EDGE_MARGIN, width - length - OBJECT_EDGE_MARGIN + 1):
            bbox = (r, c0, r, c0 + length - 1)
            if _bbox_is_free(blocked, bbox):
                boxes.append(bbox)
    random.shuffle(boxes)
    return boxes


def _place_segments(
    blocked: np.ndarray,
    lengths: list[int],
    color_cycle: list[int],
) -> list[dict] | None:
    placed_by_length: dict[int, dict] = {}
    color_of_length = {length: color for length, color in zip(lengths, color_cycle)}
    for length in sorted(lengths, reverse=True):
        candidates = _candidate_segment_bboxes(blocked, length)
        if not candidates:
            return None
        bbox = candidates[0]
        _mark_blocked(blocked, bbox)
        placed_by_length[length] = {
            "length": int(length),
            "color": int(color_of_length[length]),
            "row": int(bbox[0]),
            "col": int(bbox[1]),
            "bbox": bbox,
        }
    return [placed_by_length[length] for length in sorted(lengths)]


def _residues_visible(frame_bbox: tuple[int, int, int, int], period: int) -> bool:
    r0, c0, r1, c1 = frame_bbox
    height = int(r1 - r0 - 1)
    width = int(c1 - c0 - 1)
    rr, cc = np.indices((height, width))
    residues = set(map(int, np.unique((rr + cc) % period)))
    return residues == set(range(period))


def _instance_is_informative(grid: np.ndarray) -> bool:
    _bg, frame, segments = _extract_objects(grid)
    cycle = [int(component["color"]) for component in sorted(segments, key=_segment_length)]
    correct = _render_cycle_fill(grid, frame["bbox"], cycle, anchor_mode="local")

    if np.array_equal(correct, grid):
        return False
    if not _residues_visible(frame["bbox"], len(cycle)):
        return False

    wrong_cycles = [
        [int(component["color"]) for component in sorted(segments, key=_segment_length, reverse=True)],
        _row_major_cycle(segments),
        _color_id_cycle(segments),
    ]
    for wrong_cycle in wrong_cycles:
        if np.array_equal(correct, _render_cycle_fill(grid, frame["bbox"], wrong_cycle, anchor_mode="local")):
            return False

    for wrong_anchor in ("global", "outer_bbox"):
        if np.array_equal(correct, _render_cycle_fill(grid, frame["bbox"], cycle, anchor_mode=wrong_anchor)):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_BOUNDS)
    width = random.randint(*GRID_SIDE_BOUNDS)

    segment_count = random.randint(*SEGMENT_COUNT_BOUNDS)
    interior_h = random.randint(*FRAME_INTERIOR_H_BOUNDS)
    interior_w = random.randint(*FRAME_INTERIOR_W_BOUNDS)
    if interior_h + interior_w - 1 < segment_count:
        return None

    outer_h = interior_h + 2
    outer_w = interior_w + 2
    frame_bbox = _sample_frame_bbox(height, width, outer_h, outer_w, segment_count)
    if frame_bbox is None:
        return None

    max_length = min(width - 2 * OBJECT_EDGE_MARGIN - 1, 10)
    if max_length < SEGMENT_LENGTH_MIN + segment_count - 1:
        return None
    lengths = sorted(random.sample(range(SEGMENT_LENGTH_MIN, max_length + 1), segment_count))

    bg, frame_color, color_cycle = _sample_color_cycle(segment_count)
    blocked = np.zeros((height, width), dtype=bool)
    _mark_blocked(blocked, frame_bbox)
    placed_segments = _place_segments(blocked, lengths, color_cycle)
    if placed_segments is None:
        return None

    row_major_lengths = [int(segment["length"]) for segment in sorted(placed_segments, key=lambda seg: (seg["row"], seg["col"]))]
    if row_major_lengths == lengths or row_major_lengths == list(reversed(lengths)):
        return None

    grid = _build_input_grid((height, width), frame_bbox, frame_color, placed_segments, bg)
    if not _instance_is_informative(grid):
        return None
    if not np.array_equal(main(grid), _render_cycle_fill(grid, frame_bbox, color_cycle, anchor_mode="local")):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid length-weave input")
