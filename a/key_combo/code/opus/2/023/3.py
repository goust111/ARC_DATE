from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, colors_present, get_bg


BBox = tuple[int, int, int, int]


def _ring_coords(box: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    coords: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _apply_transform(grid: np.ndarray, code: int) -> np.ndarray:
    code = int(code) % 8
    if code == 0:
        return grid.copy()
    if code == 1:
        return np.rot90(grid, 1).copy()
    if code == 2:
        return np.rot90(grid, 2).copy()
    if code == 3:
        return np.rot90(grid, 3).copy()
    if code == 4:
        return np.flipud(grid).copy()
    if code == 5:
        return np.fliplr(grid).copy()
    if code == 6:
        return np.transpose(grid).copy()
    return np.fliplr(np.transpose(grid)).copy()


def _frame_data(grid: np.ndarray) -> tuple[int, dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    frames: dict[int, dict[str, object]] = {}
    for color in colors_present(g, exclude={bg}):
        vis = {tuple(map(int, rc)) for rc in np.argwhere(g == int(color))}
        box = bbox(g == int(color))
        assert box is not None, "each frame color must be visible"
        ring = _ring_coords(box)
        assert vis <= ring, "visible pixels must lie on the reconstructed frame border"
        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5, "frames must be at least 5x5"
        assert any(r == r0 for r, _c in vis), "top edge must stay visible"
        assert any(r == r1 for r, _c in vis), "bottom edge must stay visible"
        assert any(c == c0 for _r, c in vis), "left edge must stay visible"
        assert any(c == c1 for _r, c in vis), "right edge must stay visible"
        frames[int(color)] = {"bbox": box, "ring": ring, "visible": vis}

    assert 3 <= len(frames) <= 5, "expected 3 to 5 frames"
    return bg, frames


def _top_order(grid: np.ndarray, frames: dict[int, dict[str, object]]) -> list[int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    above: dict[int, set[int]] = {color: set() for color in frames}
    indegree: dict[int, int] = {color: 0 for color in frames}

    for lower, info in frames.items():
        ring = info["ring"]
        assert isinstance(ring, set)
        for coord in ring:
            seen = int(g[coord])
            assert seen != bg, "every reconstructed border cell must be visible through some frame"
            if seen != int(lower):
                assert seen in frames, "only frame colors may appear on a frame border"
                if int(lower) not in above[seen]:
                    above[seen].add(int(lower))
                    indegree[int(lower)] += 1

    ready = [color for color, deg in indegree.items() if deg == 0]
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "occlusion order must be uniquely linear"
        top = ready.pop()
        order.append(int(top))
        for lower in sorted(above[top]):
            indegree[lower] -= 1
            if indegree[lower] == 0:
                ready.append(int(lower))

    assert len(order) == len(frames), "occlusion graph must be acyclic"
    return order


def _successor_map(order: list[int]) -> dict[int, int]:
    return {int(order[i]): int(order[(i + 1) % len(order)]) for i in range(len(order))}


def _occupants_at(
    cell: tuple[int, int], order: list[int], frames: dict[int, dict[str, object]]
) -> list[int]:
    owners: list[int] = []
    for color in order:
        ring = frames[color]["ring"]
        assert isinstance(ring, set)
        if cell in ring:
            owners.append(int(color))
    return owners


def _expected_overlap_recolor(
    grid: np.ndarray, order: list[int], frames: dict[int, dict[str, object]]
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    succ = _successor_map(order)
    bg = int(get_bg(g))
    for r_raw, c_raw in np.argwhere(g != bg):
        cell = (int(r_raw), int(c_raw))
        owners = _occupants_at(cell, order, frames)
        assert owners, "every visible border pixel must belong to at least one frame"
        assert owners[0] == int(g[cell]), "visible pixel must come from the topmost owning frame"
        if len(owners) >= 2:
            directly_covered = owners[1]
            out[cell] = succ[directly_covered]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each visible overlap site of stacked hollow rectangles with the cyclic successor of the frame directly underneath it.

    Input structure:
    - The background is black.
    - There are 3 to 5 differently colored one-pixel-thick hollow rectangular frames.
    - Each frame may be partially hidden by higher frames, but the bbox of that color's visible pixels is still the full frame bbox.

    Rule:
    - Reconstruct every frame as the border of its visible-color bbox.
    - Infer the unique top-to-bottom occlusion order: if a cell on frame A's reconstructed border visibly shows frame B's color, then B is above A.
    - Build the cyclic successor in that depth order: top -> second -> ... -> bottom -> top.
    - For each visible border pixel, list all reconstructed frames whose borders contain that cell. If only one frame owns it, keep the pixel unchanged.
    - If several frames own the same visible cell, the top owner is the color already shown in the input; recolor that cell with the cyclic successor of the next frame directly underneath it.

    Key insight:
    - The replacement color is controlled by depth order, not by spatial order or by the visible cover color alone: you must recover which frame sits immediately under each overlap before the cyclic recoloring becomes determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, frames = _frame_data(g)
    order = _top_order(g, frames)
    return _expected_overlap_recolor(g, order, frames)


# --------------- generator ---------------

_FRAME_COUNT_CYCLE = (3, 4, 5, 4)
_STEP_CHOICES = ((2, 4), (4, 2), (3, 4), (4, 3))
_SIZE_RANGE = (5, 8)
_PALETTE = tuple(Color.NOT_BLACK)
_PADDING_RANGE = (1, 4)
_GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _sample_boxes(frame_count: int) -> list[BBox] | None:
    dr, dc = random.choice(_STEP_CHOICES)
    heights = [random.randint(*_SIZE_RANGE) for _ in range(frame_count)]
    widths = [random.randint(*_SIZE_RANGE) for _ in range(frame_count)]
    boxes = [
        (i * dr, i * dc, i * dr + heights[i] - 1, i * dc + widths[i] - 1)
        for i in range(frame_count)
    ]
    rings = [_ring_coords(box) for box in boxes]
    for i in range(frame_count - 1):
        if len(rings[i] & rings[i + 1]) < 2:
            return None
    for i in range(frame_count):
        for j in range(i + 2, frame_count):
            if rings[i] & rings[j]:
                return None
    return boxes


def _shift_boxes(boxes: list[BBox]) -> tuple[list[BBox], tuple[int, int]]:
    max_r = max(box[2] for box in boxes) + 1
    max_c = max(box[3] for box in boxes) + 1
    top = random.randint(*_PADDING_RANGE)
    left = random.randint(*_PADDING_RANGE)
    bottom = random.randint(*_PADDING_RANGE)
    right = random.randint(*_PADDING_RANGE)
    height = max_r + top + bottom
    width = max_c + left + right
    if height > 30 or width > 30:
        raise ValueError("shifted layout exceeds ARC size bound")
    shifted = [
        (r0 + top, c0 + left, r1 + top, c1 + left)
        for r0, c0, r1, c1 in boxes
    ]
    return shifted, (height, width)


def _render_input(shape: tuple[int, int], order: list[int], boxes_by_color: dict[int, BBox]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in reversed(order):
        _draw_frame(grid, boxes_by_color[color], color)
    return grid


def _model_expected(
    inp: np.ndarray, order: list[int], boxes_by_color: dict[int, BBox]
) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    frames = {
        int(color): {"bbox": box, "ring": _ring_coords(box), "visible": set()}
        for color, box in boxes_by_color.items()
    }
    hidden: dict[int, set[tuple[int, int]]] = {}
    for color, info in frames.items():
        ring = info["ring"]
        assert isinstance(ring, set)
        visible = {cell for cell in ring if int(inp[cell]) == int(color)}
        info["visible"] = visible
        hidden[int(color)] = ring - visible
    return _expected_overlap_recolor(inp, order, frames), hidden


def _valid_sample(inp: np.ndarray, expected: np.ndarray, order: list[int]) -> bool:
    try:
        bg, frames = _frame_data(inp)
        if bg != Color.BLACK:
            return False
        inferred = _top_order(inp, frames)
    except Exception:
        return False

    if inferred != order:
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    succ = _successor_map(order)
    changed_cells = 0
    witnessed_frames: set[int] = set()
    for r_raw, c_raw in np.argwhere(inp != 0):
        cell = (int(r_raw), int(c_raw))
        if int(inp[cell]) == int(expected[cell]):
            continue
        owners = _occupants_at(cell, order, frames)
        if len(owners) < 2:
            return False
        covered = owners[1]
        if int(expected[cell]) != succ[covered]:
            return False
        witnessed_frames.add(int(covered))
        changed_cells += 1

    if changed_cells < len(order) - 1:
        return False
    if witnessed_frames != set(order[1:]):
        return False

    return True


def _gen_attempt(target_count: int) -> np.ndarray | None:
    boxes = _sample_boxes(target_count)
    if boxes is None:
        return None
    try:
        shifted_boxes, shape = _shift_boxes(boxes)
    except ValueError:
        return None

    order = random.sample(list(_PALETTE), target_count)
    boxes_by_color = {int(color): box for color, box in zip(order, shifted_boxes)}
    inp = _render_input(shape, order, boxes_by_color)
    expected, hidden = _model_expected(inp, order, boxes_by_color)

    transform = (_GEN_CALL_COUNT + random.randrange(8)) % 8
    inp = _apply_transform(inp, transform)
    expected = _apply_transform(expected, transform)

    if not _valid_sample(inp, expected, order):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    target_count = _FRAME_COUNT_CYCLE[_GEN_CALL_COUNT % len(_FRAME_COUNT_CYCLE)]
    _GEN_CALL_COUNT += 1
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(target_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlap-chain frame puzzle")
