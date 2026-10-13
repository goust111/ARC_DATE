from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, label_mask


SideSet = tuple[str, ...]
BBox = tuple[int, int, int, int]
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL]
SIDE_SETS: list[SideSet] = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
    ("top", "bottom", "left"),
    ("top", "bottom", "right"),
    ("top", "left", "right"),
    ("bottom", "left", "right"),
]
GEN_ATTEMPTS = 500


def _box_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _box_dims(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _add_box(box: BBox, step: BBox, sign: int = 1) -> BBox:
    return tuple(int(box[i] + sign * step[i]) for i in range(4))


def _draw_sides(shape: tuple[int, int], box: BBox, sides: SideSet) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    if r0 > r1 or c0 > c1:
        return mask

    cc0 = max(0, c0)
    cc1 = min(w - 1, c1)
    if cc0 <= cc1:
        if "top" in sides and 0 <= r0 < h:
            mask[r0, cc0 : cc1 + 1] = True
        if "bottom" in sides and 0 <= r1 < h:
            mask[r1, cc0 : cc1 + 1] = True

    rr0 = max(0, r0)
    rr1 = min(h - 1, r1)
    if rr0 <= rr1:
        if "left" in sides and 0 <= c0 < w:
            mask[rr0 : rr1 + 1, c0] = True
        if "right" in sides and 0 <= c1 < w:
            mask[rr0 : rr1 + 1, c1] = True
    return mask


def _box_mask(shape: tuple[int, int], box: BBox, *, strict: bool) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = map(int, box)
    if strict:
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    mask = np.zeros(shape, dtype=bool)
    if rr0 <= rr1 and cc0 <= cc1:
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _strictly_contains(box: BBox, cell: tuple[int, int]) -> bool:
    r, c = map(int, cell)
    r0, c0, r1, c1 = map(int, box)
    return r0 < r < r1 and c0 < c < c1


def _component_sides(shape: tuple[int, int], coords: np.ndarray, box: BBox) -> SideSet:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    r0, c0, r1, c1 = map(int, box)
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


def _exact_full_component(shape: tuple[int, int], coords: np.ndarray, box: BBox, sides: SideSet) -> bool:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return np.array_equal(mask, _draw_sides(shape, box, sides))


def _connected(mask: np.ndarray) -> bool:
    _lab, n = label_mask(mask, connectivity=4)
    return int(n) == 1


def _regions_from_boxes(shape: tuple[int, int], boxes: list[BBox]) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    stroke_mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        stroke_mask |= _draw_sides(shape, box, _regions_from_boxes.sides)  # type: ignore[attr-defined]

    regions: list[np.ndarray] = []
    prev_full = None
    for box in boxes:
        strict = _box_mask(shape, box, strict=True)
        region = strict if prev_full is None else strict & ~prev_full
        region &= ~stroke_mask
        regions.append(region)
        prev_full = _box_mask(shape, box, strict=False)

    outer = ~_box_mask(shape, boxes[-1], strict=False)
    outer &= ~stroke_mask
    regions.append(outer)
    areas = [int(region.sum()) for region in regions]
    return stroke_mask, regions, areas


def _render_output(shape: tuple[int, int], boxes: list[BBox], sides: SideSet) -> np.ndarray:
    _regions_from_boxes.sides = sides  # type: ignore[attr-defined]
    stroke_mask, regions, areas = _regions_from_boxes(shape, boxes)
    assert len(regions) <= len(PALETTE), "too many regions for the fixed recolor palette"
    assert min(areas) > 0, "all regions must be non-empty"
    assert len(set(areas)) == len(areas), "region areas must be unique for an unambiguous ranking"

    out = np.zeros(shape, dtype=int)
    out[stroke_mask] = Color.GREY
    for rank, idx in enumerate(np.argsort(areas)):
        out[regions[int(idx)]] = PALETTE[rank]
    return out


def _infer_family(grid: np.ndarray) -> tuple[list[BBox], SideSet, tuple[int, int]]:
    h, w = grid.shape
    marker_coords = np.argwhere((grid != Color.BLACK) & (grid != Color.GREY))
    assert marker_coords.shape == (1, 2), "expected exactly one non-grey marker pixel"
    marker = tuple(map(int, marker_coords[0]))

    comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
        return_format="dict",
    )
    assert 2 <= len(comps) <= 3, "expected two or three observed grey strokes"

    full_boxes: list[BBox] = []
    full_sides: list[SideSet] = []
    accounted = np.zeros(grid.shape, dtype=bool)
    grey_mask = grid == Color.GREY

    for comp in comps:
        coords = comp["coords"]
        box = tuple(map(int, comp["bbox"]))
        sides = _component_sides(grid.shape, coords, box)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        accounted |= mask
        if len(sides) < 2:
            continue
        if not (0 < box[0] and 0 < box[1] and box[2] < h - 1 and box[3] < w - 1):
            continue
        if _exact_full_component(grid.shape, coords, box, sides):
            full_boxes.append(box)
            full_sides.append(sides)

    assert len(full_boxes) >= 2, "expected at least two fully visible strokes"
    assert len({tuple(s) for s in full_sides}) == 1, "all full strokes must use the same side-set"
    sides = full_sides[0]

    full_boxes.sort(key=_box_area)
    deltas = [
        tuple(int(full_boxes[i + 1][j] - full_boxes[i][j]) for j in range(4))
        for i in range(len(full_boxes) - 1)
    ]
    assert len({delta for delta in deltas}) == 1, "fully visible strokes must be consecutive arithmetic steps"
    step = deltas[0]
    s = int(-step[0])
    assert s > 0 and step == (-s, -s, s, s), "boxes must stay center-aligned and grow equally on all sides"

    smallest = full_boxes[0]
    while True:
        prev_box = _add_box(smallest, step, sign=-1)
        prev_h, prev_w = _box_dims(prev_box)
        if prev_h < 3 or prev_w < 3:
            break
        if _strictly_contains(prev_box, marker):
            smallest = prev_box
        else:
            break

    observed_boxes = list(full_boxes)
    observed_mask = np.zeros(grid.shape, dtype=bool)
    for box in observed_boxes:
        observed_mask |= _draw_sides(grid.shape, box, sides)

    remaining = grey_mask & ~observed_mask
    last = observed_boxes[-1]
    while remaining.any():
        last = _add_box(last, step, sign=1)
        next_mask = _draw_sides(grid.shape, last, sides)
        assert np.any(next_mask), "remaining pixels cannot belong to a fully off-grid stroke"
        assert np.array_equal(remaining, next_mask), "any non-full observed stroke must be the next arithmetic member"
        observed_boxes.append(last)
        observed_mask |= next_mask
        remaining = grey_mask & ~observed_mask

    assert np.array_equal(observed_mask, grey_mask), "observed grey pixels must be explained by the inferred family"

    all_boxes: list[BBox] = []
    cur = smallest
    limit = h + w + 10
    for _ in range(limit):
        mask = _draw_sides(grid.shape, cur, sides)
        if not np.any(mask):
            break
        all_boxes.append(cur)
        cur = _add_box(cur, step, sign=1)
    else:
        raise AssertionError("stroke family did not leave the grid")

    assert _strictly_contains(all_boxes[0], marker), "marker must lie inside the innermost completed rectangle"
    return all_boxes, sides, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the arithmetic family of clipped rectangle side-strokes, then recolor the nested rectangular regions by area.

    Input structure:
    - A black grid contains two or three observed grey rectangle side-strokes, all using the same 2-side corner or 3-side U pattern.
    - The stroke bounding boxes are center-aligned and grow by one constant amount on all four sides from one family member to the next.
    - Some family members are omitted from the input, and larger members may be clipped by the canvas boundary.
    - A single non-black, non-grey marker pixel lies inside the true innermost rectangle region.

    Rule:
    - Infer the arithmetic bbox step from the fully visible observed strokes.
    - Extrapolate inward until the next smaller rectangle would no longer strictly contain the marker pixel.
    - Extrapolate outward until the next larger stroke has no visible in-bounds pixels.
    - Draw every inferred grey stroke, treat the completed nested rectangles as full rectangular partitions, and compute the pixel area of each region excluding stroke pixels.
    - Sort the regions by increasing area and recolor them blue, red, green, yellow, pink, orange, teal in that order.

    Key insight:
    - The grey pixels only reveal a few consecutive members of a larger center-aligned arithmetic rectangle family; the marker fixes the innermost missing member, and only the completed family reveals the correct set of nested regions to rank by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9, "expected ARC palette values"

    boxes, sides, _marker = _infer_family(g)
    return _render_output(g.shape, boxes, sides)


def _sample_box_family() -> tuple[tuple[int, int], list[BBox], SideSet, int] | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    sides = random.choice(SIDE_SETS)
    step_size = 2
    base_h = random.choice([5, 7, 9])
    base_w = random.choice([5, 7, 9])
    center_r = random.randint(3, h - 4)
    center_c = random.randint(3, w - 4)

    half_h = (base_h - 1) // 2
    half_w = (base_w - 1) // 2
    box0 = (center_r - half_h, center_c - half_w, center_r + half_h, center_c + half_w)

    boxes: list[BBox] = []
    cur = box0
    for _ in range(10):
        mask = _draw_sides((h, w), cur, sides)
        if not np.any(mask):
            break
        boxes.append(cur)
        cur = _add_box(cur, (-step_size, -step_size, step_size, step_size), sign=1)

    if len(boxes) not in (4, 5, 6):
        return None
    return (h, w), boxes, sides, step_size


def _marker_candidates(shape: tuple[int, int], smallest: BBox, step_size: int) -> list[tuple[int, int]]:
    inner = _box_mask(shape, smallest, strict=True)
    prev_box = _add_box(smallest, (-step_size, -step_size, step_size, step_size), sign=-1)
    prev_h, prev_w = _box_dims(prev_box)
    if prev_h >= 3 and prev_w >= 3:
        inner &= ~_box_mask(shape, prev_box, strict=True)
    return [tuple(map(int, rc)) for rc in np.argwhere(inner)]


def _input_from_design(
    shape: tuple[int, int],
    boxes: list[BBox],
    sides: SideSet,
    observed_start: int,
    observed_len: int,
    marker: tuple[int, int],
    marker_color: int,
) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    for box in boxes[observed_start : observed_start + observed_len]:
        inp[_draw_sides(shape, box, sides)] = Color.GREY
    inp[marker] = int(marker_color)
    return inp


def _valid_observed_slice(
    shape: tuple[int, int],
    boxes: list[BBox],
    sides: SideSet,
    observed_start: int,
    observed_len: int,
) -> bool:
    observed = boxes[observed_start : observed_start + observed_len]
    if len(observed) != observed_len:
        return False
    for box in observed[:2]:
        r0, c0, r1, c1 = box
        if not (0 < r0 and 0 < c0 and r1 < shape[0] - 1 and c1 < shape[1] - 1):
            return False
    for box in observed:
        if not _connected(_draw_sides(shape, box, sides)):
            return False
    return True


def _semantic_checks(
    inp: np.ndarray,
    boxes: list[BBox],
    sides: SideSet,
    observed_start: int,
    observed_len: int,
    marker: tuple[int, int],
) -> None:
    grey_mask = inp == Color.GREY
    observed_mask = np.zeros(inp.shape, dtype=bool)
    for box in boxes[observed_start : observed_start + observed_len]:
        observed_mask |= _draw_sides(inp.shape, box, sides)
    assert np.array_equal(grey_mask, observed_mask), "input must show exactly the chosen observed strokes"
    assert observed_start >= 1, "at least one smaller family member must be hidden"
    assert observed_start + observed_len < len(boxes), "at least one larger family member must be hidden"
    assert _strictly_contains(boxes[0], marker), "marker must sit inside the true innermost region"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sampled = _sample_box_family()
        if sampled is None:
            continue
        shape, boxes, sides, step_size = sampled

        observed_len = 2 if len(boxes) == 4 else random.choice([2, 3])
        max_start = len(boxes) - observed_len - 1
        observed_start = random.randint(1, max_start)
        if not _valid_observed_slice(shape, boxes, sides, observed_start, observed_len):
            continue

        marker_options = _marker_candidates(shape, boxes[0], step_size)
        if not marker_options:
            continue
        marker = random.choice(marker_options)
        marker_color = random.choice([Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON])

        try:
            expected = _render_output(shape, boxes, sides)
        except AssertionError:
            continue
        inp = _input_from_design(shape, boxes, sides, observed_start, observed_len, marker, marker_color)

        try:
            _semantic_checks(inp, boxes, sides, observed_start, observed_len, marker)
            solved = main(inp)
        except AssertionError:
            continue

        if not np.array_equal(solved, expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid nested-rectangle compression input")
