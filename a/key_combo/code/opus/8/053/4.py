from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
TEAL = 8
_FRAME_PALETTE = (1, 2, 3, 4, 7)
_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_BASE_TEMPLATES = {
    2: np.array(
        [
            [TEAL, BLACK, BLACK],
            [TEAL, BLACK, BLACK],
            [TEAL, BLACK, BLACK],
        ],
        dtype=int,
    ),
    3: np.array(
        [
            [BLACK, BLACK, BLACK],
            [TEAL, TEAL, TEAL],
            [BLACK, BLACK, BLACK],
        ],
        dtype=int,
    ),
    4: np.array(
        [
            [TEAL, BLACK, BLACK],
            [BLACK, TEAL, BLACK],
            [BLACK, BLACK, TEAL],
        ],
        dtype=int,
    ),
}

_ROTATION_K = {
    "center": 0,
    "up": 0,
    "right": 3,
    "down": 2,
    "left": 1,
}

_SCHEDULE = [
    (2, "left"),
    (3, "down"),
    (4, "up"),
    (2, "right"),
    (3, "center"),
    (4, "left"),
    (2, "up"),
    (3, "right"),
    (4, "down"),
    (2, "center"),
    (3, "left"),
    (4, "right"),
    (2, "down"),
    (3, "up"),
    (4, "center"),
]
_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_rectangular_ring(comp: dict) -> bool:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return False

    patch = np.zeros((h, w), dtype=bool)
    local = coords - np.array([r0, c0])
    patch[local[:, 0], local[:, 1]] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(patch, border))


def _direction_from_pair(outer_box: tuple[int, int, int, int], inner_box: tuple[int, int, int, int]) -> str:
    or0, oc0, or1, oc1 = outer_box
    ir0, ic0, ir1, ic1 = inner_box

    top_gap = ir0 - or0 - 1
    bottom_gap = or1 - ir1 - 1
    left_gap = ic0 - oc0 - 1
    right_gap = oc1 - ic1 - 1

    assert min(top_gap, bottom_gap, left_gap, right_gap) >= 1, "adjacent frames must be separated by black space"

    horizontal = 0
    if left_gap > right_gap:
        horizontal = 1
    elif left_gap < right_gap:
        horizontal = -1

    vertical = 0
    if top_gap > bottom_gap:
        vertical = 1
    elif top_gap < bottom_gap:
        vertical = -1

    assert not (horizontal and vertical), "innermost frame may shift along at most one axis"

    if horizontal > 0:
        return "right"
    if horizontal < 0:
        return "left"
    if vertical > 0:
        return "down"
    if vertical < 0:
        return "up"
    return "center"


def _extract_layers(grid: np.ndarray) -> tuple[list[dict], str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    comps = _find_components(g, bg)
    assert 2 <= len(comps) <= 4, "expected 2 to 4 nested frames"
    assert len({comp["color"] for comp in comps}) == len(comps), "each frame must use its own color"

    for comp in comps:
        assert _is_rectangular_ring(comp), "every component must be a 1-pixel rectangular ring"

    layers = sorted(comps, key=lambda comp: ((comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1)), reverse=True)

    for outer, inner in zip(layers, layers[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1, "frames must be strictly nested"
        assert ir0 - or0 >= 2 and ic0 - oc0 >= 2 and or1 - ir1 >= 2 and oc1 - ic1 >= 2, (
            "adjacent frames need at least one black row and column between them"
        )

    for outer, inner in zip(layers[:-2], layers[1:-1]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert ir0 - or0 == or1 - ir1, "all non-final nested pairs must stay vertically concentric"
        assert ic0 - oc0 == oc1 - ic1, "all non-final nested pairs must stay horizontally concentric"

    direction = _direction_from_pair(layers[-2]["bbox"], layers[-1]["bbox"])
    return layers, direction


def _output_from_code(depth: int, direction: str) -> np.ndarray:
    assert depth in _BASE_TEMPLATES, "unsupported nesting depth"
    base = _BASE_TEMPLATES[depth]
    return np.rot90(base, _ROTATION_K[direction]).astype(int, copy=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the nested rectangular-frame depth into a fixed 3x3 teal template, then rotate it by the innermost frame's eccentric direction.

    Input structure:
    - The grid has a black background and exactly one group of 2 to 4 strictly nested one-pixel rectangular frames.
    - Each frame is monochromatic, each adjacent pair is separated by at least one black row and one black column, and all outer pairs stay concentric.
    - The innermost frame encloses black and may be centered, shifted left, shifted right, shifted up, or shifted down relative to the next outer frame.

    Rule:
    - Identify all nested rectangular-frame layers and count them.
    - Use the depth as a lookup key: depth 2 gives a left-column teal bar, depth 3 gives a middle-row teal bar, and depth 4 gives a main-diagonal teal pattern on a 3x3 black grid.
    - Compare the innermost frame with the next outer frame: centered or up keeps the template unchanged, right rotates it 90 degrees clockwise, down rotates it 180 degrees, and left rotates it 90 degrees counterclockwise.

    Key insight:
    - The output class is not encoded by any particular color or size; it comes from the number of nested rectangular rings, while the innermost ring's off-center placement supplies an independent rotation signal.
    """
    layers, direction = _extract_layers(np.asarray(grid, dtype=int))
    return _output_from_code(len(layers), direction)


def _expand_box(
    inner_box: tuple[int, int, int, int],
    gaps: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    top_gap, bottom_gap, left_gap, right_gap = gaps
    ir0, ic0, ir1, ic1 = inner_box
    return (
        ir0 - (1 + top_gap),
        ic0 - (1 + left_gap),
        ir1 + (1 + bottom_gap),
        ic1 + (1 + right_gap),
    )


def _choose_innermost_gaps(direction: str) -> tuple[int, int, int, int]:
    if direction == "left":
        vertical = random.randint(1, 2)
        return vertical, vertical, 1, 3
    if direction == "right":
        vertical = random.randint(1, 2)
        return vertical, vertical, 3, 1
    if direction == "up":
        horizontal = random.randint(1, 2)
        return 1, 3, horizontal, horizontal
    if direction == "down":
        horizontal = random.randint(1, 2)
        return 3, 1, horizontal, horizontal
    vertical = random.randint(1, 2)
    horizontal = random.randint(1, 2)
    return vertical, vertical, horizontal, horizontal


def _draw_ring(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _color_sequence(depth: int) -> tuple[int, ...]:
    return tuple(sorted(random.sample(_FRAME_PALETTE, depth), key=_FRAME_PALETTE.index))


def _make_boxes(depth: int, direction: str) -> list[tuple[int, int, int, int]]:
    inner_h = random.randint(3, 7)
    inner_w = random.randint(3, 7)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]

    boxes.append(_expand_box(boxes[-1], _choose_innermost_gaps(direction)))
    while len(boxes) < depth:
        vertical = random.randint(1, 2)
        horizontal = random.randint(1, 2)
        boxes.append(_expand_box(boxes[-1], (vertical, vertical, horizontal, horizontal)))

    r_shift = -min(box[0] for box in boxes)
    c_shift = -min(box[1] for box in boxes)
    normalized = []
    for r0, c0, r1, c1 in boxes:
        normalized.append((r0 + r_shift, c0 + c_shift, r1 + r_shift, c1 + c_shift))
    normalized.reverse()
    return normalized


def _render_candidate(depth: int, direction: str) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    boxes = _make_boxes(depth, direction)
    colors = _color_sequence(depth)
    outer = boxes[0]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    grid = np.zeros((outer_h + top_margin + bottom_margin, outer_w + left_margin + right_margin), dtype=int)

    shifted_boxes = []
    for box, color in zip(boxes, colors):
        r0, c0, r1, c1 = box
        shifted = (
            r0 + top_margin,
            c0 + left_margin,
            r1 + top_margin,
            c1 + left_margin,
        )
        shifted_boxes.append(shifted)
        _draw_ring(grid, shifted, color)

    return grid, shifted_boxes


def _validate_generator_grid(
    grid: np.ndarray,
    depth: int,
    direction: str,
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    if not np.array_equal(main(grid), _output_from_code(depth, direction)):
        return False

    try:
        layers, parsed_direction = _extract_layers(grid)
    except AssertionError:
        return False

    if len(layers) != depth or parsed_direction != direction:
        return False

    parsed_boxes = [layer["bbox"] for layer in layers]
    if parsed_boxes != boxes:
        return False

    for outer, inner in zip(boxes[:-1], boxes[1:]):
        or0, oc0, or1, oc1 = outer
        ir0, ic0, ir1, ic1 = inner
        if min(ir0 - or0 - 1, ic0 - oc0 - 1, or1 - ir1 - 1, oc1 - ic1 - 1) < 1:
            return False

    if np.count_nonzero(grid) != sum(2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4 for r0, c0, r1, c1 in boxes):
        return False

    return True


def _scheduled_combo() -> tuple[int, str]:
    global _GEN_CALL_COUNT
    combo = _SCHEDULE[_GEN_CALL_COUNT % len(_SCHEDULE)]
    _GEN_CALL_COUNT += 1
    return combo


def generate_input() -> np.ndarray:
    depth, direction = _scheduled_combo()
    for _ in range(400):
        grid, boxes = _render_candidate(depth, direction)
        if _validate_generator_grid(grid, depth, direction, boxes):
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
