from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg
import random


Side = str
BBox = tuple[int, int, int, int]


def _component_sides(shape: tuple[int, int], coords: np.ndarray, box: BBox) -> tuple[Side, ...]:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    r0, c0, r1, c1 = box

    sides = []
    if np.all(mask[r0, c0 : c1 + 1]):
        sides.append("top")
    if np.all(mask[r1, c0 : c1 + 1]):
        sides.append("bottom")
    if np.all(mask[r0 : r1 + 1, c0]):
        sides.append("left")
    if np.all(mask[r0 : r1 + 1, c1]):
        sides.append("right")
    return tuple(sides)


def _draw_clipped_sides(shape: tuple[int, int], box: BBox, sides: tuple[Side, ...]) -> np.ndarray:
    height, width = shape
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    if r0 > r1 or c0 > c1:
        return mask

    clip_c0 = max(0, c0)
    clip_c1 = min(width - 1, c1)
    if clip_c0 <= clip_c1:
        if "top" in sides and 0 <= r0 < height:
            mask[r0, clip_c0 : clip_c1 + 1] = True
        if "bottom" in sides and 0 <= r1 < height:
            mask[r1, clip_c0 : clip_c1 + 1] = True

    clip_r0 = max(0, r0)
    clip_r1 = min(height - 1, r1)
    if clip_r0 <= clip_r1:
        if "left" in sides and 0 <= c0 < width:
            mask[clip_r0 : clip_r1 + 1, c0] = True
        if "right" in sides and 0 <= c1 < width:
            mask[clip_r0 : clip_r1 + 1, c1] = True

    return mask


def _box_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _add_box(box: BBox, delta: BBox, sign: int = 1) -> BBox:
    return tuple(int(box[i] + sign * delta[i]) for i in range(4))


def _extend_boxes(
    shape: tuple[int, int],
    boxes: list[BBox],
    step: BBox,
    sides: tuple[Side, ...],
) -> list[BBox]:
    all_boxes = list(boxes)
    limit = 2 * (shape[0] + shape[1]) + 10

    cur = boxes[-1]
    for _ in range(limit):
        cur = _add_box(cur, step)
        if not np.any(_draw_clipped_sides(shape, cur, sides)):
            break
        all_boxes.append(cur)
    else:
        raise AssertionError("forward stroke sequence did not leave the grid")

    cur = boxes[0]
    for _ in range(limit):
        cur = _add_box(cur, step, sign=-1)
        if not np.any(_draw_clipped_sides(shape, cur, sides)):
            break
        all_boxes.append(cur)
    else:
        raise AssertionError("backward stroke sequence did not leave the grid")

    return all_boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the arithmetic sequence of rectangular side-strokes and turn all remaining background grey.

    Input structure
    A zero-background 2D grid contains one non-background color. Its 4-connected
    objects are clipped rectangular side-strokes: each object occupies the same
    set of bbox sides, such as top+right, left+bottom, or a three-sided U.

    Rule
    Sort the observed stroke bboxes by increasing bbox area. Consecutive bboxes
    differ by a constant coordinate delta, so extend that bbox sequence in both
    directions until the corresponding clipped side-stroke has no visible pixels.
    Paint every visible side-stroke with the input foreground color and paint
    all other cells grey.

    Key insight
    The missing output pixels are not filled from local adjacency; the visible
    objects are consecutive members of one larger arithmetic family of rectangular
    side-strokes, with larger members clipped by the canvas boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected one foreground color"
    fg = int(foreground_colors[0])
    assert fg != Color.GREY, "grey is reserved for the output fill"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) >= 2, "expected at least two visible strokes"

    sides_by_component = [
        _component_sides(g.shape, comp["coords"], comp["bbox"])
        for comp in components
    ]
    assert len(set(sides_by_component)) == 1, "all strokes must use the same bbox sides"
    sides = sides_by_component[0]
    assert len(sides) >= 2, "expected each stroke to use at least two bbox sides"

    for comp in components:
        actual = np.zeros(g.shape, dtype=bool)
        actual[comp["coords"][:, 0], comp["coords"][:, 1]] = True
        expected = _draw_clipped_sides(g.shape, comp["bbox"], sides)
        assert np.array_equal(actual, expected), "component must exactly match its bbox sides"

    boxes = [tuple(map(int, comp["bbox"])) for comp in components]
    areas = [_box_area(box) for box in boxes]
    assert len(set(areas)) == len(areas), "stroke bbox areas must be unique"
    boxes.sort(key=_box_area)
    deltas = [
        tuple(int(boxes[i + 1][j] - boxes[i][j]) for j in range(4))
        for i in range(len(boxes) - 1)
    ]
    assert len(set(deltas)) == 1, "stroke bboxes must form an arithmetic sequence"
    step = deltas[0]
    assert any(step), "stroke sequence must have a nonzero step"

    out = np.full(g.shape, Color.GREY, dtype=int)
    for box in _extend_boxes(g.shape, boxes, step, sides):
        out[_draw_clipped_sides(g.shape, box, sides)] = fg
    return out

# --------------- generator (o5x) ---------------

from arc_utils import bbox_coords


# Canvas, backgrounds, and palette.
_GEN_F8C80D96_SHAPE = (10, 10)
_GEN_F8C80D96_INPUT_BG = Color.BLACK
_GEN_F8C80D96_OUTPUT_BG = Color.GREY
_GEN_F8C80D96_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)

# Stroke family choices.
_GEN_F8C80D96_SIDE_SETS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
    ("top", "bottom", "left"),
    ("top", "bottom", "right"),
    ("top", "left", "right"),
    ("bottom", "left", "right"),
)
_GEN_F8C80D96_STEP_MAGNITUDES = (2, 3)
_GEN_F8C80D96_INPUT_STROKE_COUNTS = (2, 3)
_GEN_F8C80D96_OUTPUT_STROKE_COUNTS = range(3, 6)

# Search and retry limits.
_GEN_F8C80D96_SEQUENCE_LIMIT = 50
_GEN_F8C80D96_ATTEMPT_LIMIT = 300
_GEN_F8C80D96_CANDIDATES = None


def _gen_f8c80d96_step(sides: tuple[str, ...], magnitude: int) -> BBox:
    return (
        -magnitude if "top" in sides else 0,
        -magnitude if "left" in sides else 0,
        magnitude if "bottom" in sides else 0,
        magnitude if "right" in sides else 0,
    )


def _gen_f8c80d96_inside(box: BBox) -> bool:
    height, width = _GEN_F8C80D96_SHAPE
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width


def _gen_f8c80d96_pins_absent_sides(box: BBox, sides: tuple[str, ...]) -> bool:
    height, width = _GEN_F8C80D96_SHAPE
    r0, c0, r1, c1 = box
    return (
        ("top" in sides or r0 == 0)
        and ("bottom" in sides or r1 == height - 1)
        and ("left" in sides or c0 == 0)
        and ("right" in sides or c1 == width - 1)
    )


def _gen_f8c80d96_exact_sides(box: BBox, sides: tuple[str, ...]) -> bool:
    mask = _draw_clipped_sides(_GEN_F8C80D96_SHAPE, box, sides)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return bbox_coords(coords) == box and _component_sides(_GEN_F8C80D96_SHAPE, coords, box) == sides


def _gen_f8c80d96_valid_seed_box(
    box: BBox,
    sides: tuple[str, ...],
    magnitude: int,
) -> bool:
    r0, c0, r1, c1 = box
    if min(r1 - r0 + 1, c1 - c0 + 1) != magnitude:
        return False
    if not _gen_f8c80d96_pins_absent_sides(box, sides):
        return False
    return _gen_f8c80d96_exact_sides(box, sides)


def _gen_f8c80d96_sequence(
    box: BBox,
    step: BBox,
    sides: tuple[str, ...],
) -> list[BBox] | None:
    if np.any(_draw_clipped_sides(_GEN_F8C80D96_SHAPE, _add_box(box, step, -1), sides)):
        return None

    boxes = [box]
    cur = box
    for _ in range(_GEN_F8C80D96_SEQUENCE_LIMIT):
        cur = _add_box(cur, step)
        if not np.any(_draw_clipped_sides(_GEN_F8C80D96_SHAPE, cur, sides)):
            return boxes
        boxes.append(cur)
    return None


def _gen_f8c80d96_windows(boxes: list[BBox]) -> list[tuple[int, ...]]:
    windows = []
    fully_visible = {i for i, box in enumerate(boxes) if _gen_f8c80d96_inside(box)}
    for count in _GEN_F8C80D96_INPUT_STROKE_COUNTS:
        for start in range(len(boxes) - count + 1):
            inds = tuple(range(start, start + count))
            if all(i in fully_visible for i in inds):
                windows.append(inds)
    return windows


def _gen_f8c80d96_candidate_from_seed(
    box: BBox,
    step: BBox,
    sides: tuple[str, ...],
    magnitude: int,
) -> dict | None:
    if not _gen_f8c80d96_valid_seed_box(box, sides, magnitude):
        return None

    boxes = _gen_f8c80d96_sequence(box, step, sides)
    if boxes is None:
        return None
    if len(boxes) not in _GEN_F8C80D96_OUTPUT_STROKE_COUNTS:
        return None

    windows = _gen_f8c80d96_windows(boxes)
    if not windows:
        return None
    return {
        "sides": sides,
        "magnitude": magnitude,
        "boxes": boxes,
        "windows": windows,
    }


def _gen_f8c80d96_candidate_catalog() -> list[dict]:
    global _GEN_F8C80D96_CANDIDATES
    if _GEN_F8C80D96_CANDIDATES is not None:
        return _GEN_F8C80D96_CANDIDATES

    height, width = _GEN_F8C80D96_SHAPE
    candidates = []
    for sides in _GEN_F8C80D96_SIDE_SETS:
        for magnitude in _GEN_F8C80D96_STEP_MAGNITUDES:
            step = _gen_f8c80d96_step(sides, magnitude)
            for r0 in range(height):
                for r1 in range(r0, height):
                    for c0 in range(width):
                        for c1 in range(c0, width):
                            box = (r0, c0, r1, c1)
                            candidate = _gen_f8c80d96_candidate_from_seed(box, step, sides, magnitude)
                            if candidate is not None:
                                candidates.append(candidate)

    _GEN_F8C80D96_CANDIDATES = candidates
    return candidates


def _gen_f8c80d96_render(
    boxes: list[BBox],
    sides: tuple[str, ...],
    color: int,
    background: int,
) -> np.ndarray:
    grid = np.full(_GEN_F8C80D96_SHAPE, background, dtype=int)
    for box in boxes:
        grid[_draw_clipped_sides(_GEN_F8C80D96_SHAPE, box, sides)] = color
    return grid


def _gen_f8c80d96_select_candidate(catalog: list[dict]) -> tuple[dict, int] | None:
    if not catalog:
        return None

    sides = random.choice(_GEN_F8C80D96_SIDE_SETS)
    magnitude = random.choice(_GEN_F8C80D96_STEP_MAGNITUDES)
    pool = [c for c in catalog if c["sides"] == sides and c["magnitude"] == magnitude]
    if not pool:
        return None
    preferred_count = random.choice(_GEN_F8C80D96_INPUT_STROKE_COUNTS)
    count_pool = [c for c in pool if any(len(w) == preferred_count for w in c["windows"])]
    if count_pool:
        pool = count_pool
    return random.choice(pool), preferred_count


def _gen_f8c80d96_select_window(candidate: dict, preferred_count: int) -> tuple[int, ...]:
    windows = [w for w in candidate["windows"] if len(w) == preferred_count]
    if not windows:
        windows = candidate["windows"]
    return random.choice(windows)


def _gen_attempt() -> np.ndarray | None:
    selection = _gen_f8c80d96_select_candidate(_gen_f8c80d96_candidate_catalog())
    if selection is None:
        return None

    candidate, preferred_count = selection
    fg = int(random.choice(_GEN_F8C80D96_COLORS))
    out = _gen_f8c80d96_render(candidate["boxes"], candidate["sides"], fg, _GEN_F8C80D96_OUTPUT_BG)

    window = _gen_f8c80d96_select_window(candidate, preferred_count)
    input_boxes = [candidate["boxes"][i] for i in window]
    inp = _gen_f8c80d96_render(input_boxes, candidate["sides"], fg, _GEN_F8C80D96_INPUT_BG)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_F8C80D96_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid f8c80d96 input")
