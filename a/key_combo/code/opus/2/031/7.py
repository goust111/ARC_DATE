from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

SIDE_PATTERNS = (
    (True, False, True, False),   # top + left
    (False, True, True, False),   # bottom + left
    (True, True, True, False),    # top + bottom + left
    (True, False, False, True),   # top + right
    (False, True, False, True),   # bottom + right
    (True, True, False, True),    # top + bottom + right
    (True, False, True, True),    # top + left + right
    (False, True, True, True),    # bottom + left + right
    (True, True, True, True),     # full frame
)

GENERATION_RETRIES = 600

BBox = tuple[int, int, int, int]
SideMask = tuple[bool, bool, bool, bool]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = shape
    return 0 <= r < height and 0 <= c < width


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _side_mask_connected(side_mask: SideMask) -> bool:
    selected = [i for i, flag in enumerate(side_mask) if flag]
    if len(selected) < 2:
        return False

    corners = {
        0: {2, 3},  # top touches left/right
        1: {2, 3},  # bottom touches left/right
        2: {0, 1},  # left touches top/bottom
        3: {0, 1},  # right touches top/bottom
    }
    stack = [selected[0]]
    seen = {selected[0]}
    while stack:
        side = stack.pop()
        for other in selected:
            if other not in seen and other in corners[side]:
                seen.add(other)
                stack.append(other)
    return len(seen) == len(selected)


def _stroke_cells(bbox: BBox, side_mask: SideMask) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    top, bottom, left, right = side_mask
    cells: list[tuple[int, int]] = []
    if top:
        for c in range(c0, c1 + 1):
            cells.append((r0, c))
    if bottom:
        for c in range(c0, c1 + 1):
            cells.append((r1, c))
    if left:
        for r in range(r0, r1 + 1):
            cells.append((r, c0))
    if right:
        for r in range(r0, r1 + 1):
            cells.append((r, c1))
    return list(dict.fromkeys(cells))


def _infer_side_mask(coords: np.ndarray, bbox: BBox) -> SideMask:
    r0, c0, r1, c1 = bbox
    cells = {tuple(map(int, rc)) for rc in coords}
    top = all((r0, c) in cells for c in range(c0, c1 + 1))
    bottom = all((r1, c) in cells for c in range(c0, c1 + 1))
    left = all((r, c0) in cells for r in range(r0, r1 + 1))
    right = all((r, c1) in cells for r in range(r0, r1 + 1))
    side_mask = (top, bottom, left, right)
    assert _side_mask_connected(side_mask), f"disconnected side mask: {side_mask}"
    assert cells == set(_stroke_cells(bbox, side_mask)), "component is not a clean border stroke"
    return side_mask


def _shift_bbox(bbox: BBox, step: BBox, direction: int) -> BBox:
    return tuple(int(v + direction * dv) for v, dv in zip(bbox, step))  # type: ignore[return-value]


def _paint_stroke(out: np.ndarray, bbox: BBox, side_mask: SideMask, color: int) -> int:
    painted = 0
    for r, c in _stroke_cells(bbox, side_mask):
        if _in_bounds(out.shape, r, c):
            out[r, c] = color
            painted += 1
    return painted


def _visible_stroke_cells(shape: tuple[int, int], bbox: BBox, side_mask: SideMask) -> list[tuple[int, int]]:
    return [(r, c) for r, c in _stroke_cells(bbox, side_mask) if _in_bounds(shape, r, c)]


def _bbox_red_coords(grid: np.ndarray, bbox: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    local = np.argwhere(patch == RED)
    return [(r0 + int(rr), c0 + int(cc)) for rr, cc in local]


def _sequence_step(bboxes: list[BBox]) -> BBox:
    arr = np.array(bboxes, dtype=int)
    diffs = arr[1:] - arr[:-1]
    assert diffs.size > 0
    assert np.all(diffs == diffs[0]), "bboxes do not follow one arithmetic step"
    step = tuple(int(v) for v in diffs[0])
    assert step != (0, 0, 0, 0)
    return step  # type: ignore[return-value]


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the blue stroke with the fewest red interior marks, rebuild the arithmetic rectangle sequence, and grey-fill the remaining background.

    Input structure:
    - The grid has a black background, 4-6 blue 4-connected rectangular border strokes, and red marker pixels inside their bounding boxes.
    - Every blue object is a union of the same connected subset of rectangle sides, such as top+right or left+bottom+right.
    - After sorting the strokes by bounding-box area, their bounding boxes form an arithmetic progression.

    Rule:
    - Count the red marker pixels inside each blue stroke's bounding box and identify the unique stroke with the fewest markers.
    - Remove that stroke and its internal red markers conceptually, then redraw the full blue sequence at the missing index and extend the same bbox progression outward in both directions until a further stroke has no visible pixels in the grid.
    - Keep all non-erased red markers unchanged and recolor every remaining black background cell to grey.

    Key insight:
    - The marker count only chooses which term to erase; reconstructing the output requires recovering the shared side pattern and the bbox arithmetic step so the missing term and the clipped edge extensions can all be drawn consistently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    components = _components_mask(g == BLUE)
    assert 4 <= len(components) <= 6

    infos = []
    for coords in components:
        bbox = _bbox_coords(coords)
        side_mask = _infer_side_mask(coords, bbox)
        red_coords = _bbox_red_coords(g, bbox)
        infos.append(
            {
                "bbox": bbox,
                "side_mask": side_mask,
                "red_coords": red_coords,
                "red_count": len(red_coords),
            }
        )

    infos.sort(key=lambda info: _bbox_area(info["bbox"]))
    areas = [_bbox_area(info["bbox"]) for info in infos]
    assert all(a < b for a, b in zip(areas, areas[1:]))

    side_mask = infos[0]["side_mask"]
    assert all(info["side_mask"] == side_mask for info in infos)
    step = _sequence_step([info["bbox"] for info in infos])

    red_counts = [info["red_count"] for info in infos]
    fewest = min(red_counts)
    targets = [i for i, count in enumerate(red_counts) if count == fewest]
    assert len(targets) == 1
    erased_index = targets[0]

    out = np.zeros_like(g)
    for info in infos:
        _paint_stroke(out, info["bbox"], side_mask, BLUE)

    first_bbox = infos[0]["bbox"]
    bbox = _shift_bbox(first_bbox, step, -1)
    limit = 2 * (g.shape[0] + g.shape[1] + 5)
    for _ in range(limit):
        if _paint_stroke(out, bbox, side_mask, BLUE) == 0:
            break
        bbox = _shift_bbox(bbox, step, -1)
    else:
        raise AssertionError("smaller-direction extension did not terminate")

    last_bbox = infos[-1]["bbox"]
    bbox = _shift_bbox(last_bbox, step, 1)
    for _ in range(limit):
        if _paint_stroke(out, bbox, side_mask, BLUE) == 0:
            break
        bbox = _shift_bbox(bbox, step, 1)
    else:
        raise AssertionError("larger-direction extension did not terminate")

    for i, info in enumerate(infos):
        if i == erased_index:
            continue
        for r, c in info["red_coords"]:
            out[r, c] = RED

    out[out == BLACK] = GREY
    return out


def _canonical_bbox(index: int, r0: int, c0: int, height: int, width0: int, dr: int, dc: int, dw: int) -> BBox:
    width = width0 + index * dw
    top = r0 + index * dr
    left = c0 + index * dc
    return top, left, top + height - 1, left + width - 1


def _interior_cells(bbox: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _gen_canonical_input() -> np.ndarray | None:
    n_strokes = random.choice([4, 4, 5, 5, 6])
    height = 3 if n_strokes == 6 else random.choice([3, 4])
    dr = height + 1
    dc = random.randint(0, 2)
    dw = random.choice([1, 1, 2])

    min_width = max((n_strokes + height - 3) // (height - 2) + 2, 2 * dw + 2)
    max_width = min(10, 30 - (n_strokes - 1) * (dc + dw))
    if min_width > max_width:
        return None
    width0 = random.randint(min_width, max_width)

    max_c0 = 30 - ((n_strokes - 1) * dc + width0 + (n_strokes - 1) * dw)
    if max_c0 < 0:
        return None
    c0 = random.randint(0, max_c0)
    r0 = random.randint(2, height)

    side_mask = random.choice(SIDE_PATTERNS)
    bboxes = {
        idx: _canonical_bbox(idx, r0, c0, height, width0, dr, dc, dw)
        for idx in range(-2, n_strokes + 2)
    }

    last_visible = bboxes[n_strokes - 1]
    next_bbox = bboxes[n_strokes]
    next_next_bbox = bboxes[n_strokes + 1]

    min_h = max(last_visible[2] + 1, next_bbox[0] + 1)
    max_h = min(30, next_bbox[2], next_next_bbox[0])
    if min_h > max_h:
        return None
    height_out = random.randint(min_h, max_h)

    min_w = last_visible[3] + 1
    max_w = min(30, min_w + 6)
    if min_w > max_w:
        return None
    width_out = random.randint(min_w, max_w)
    shape = (height_out, width_out)

    prev_visible = _visible_stroke_cells(shape, bboxes[-1], side_mask)
    next_visible = _visible_stroke_cells(shape, bboxes[n_strokes], side_mask)
    if not prev_visible or not next_visible:
        return None
    if len(prev_visible) == len(_stroke_cells(bboxes[-1], side_mask)):
        return None
    if len(next_visible) == len(_stroke_cells(bboxes[n_strokes], side_mask)):
        return None
    if _visible_stroke_cells(shape, bboxes[-2], side_mask):
        return None
    if _visible_stroke_cells(shape, bboxes[n_strokes + 1], side_mask):
        return None

    grid = np.zeros(shape, dtype=int)
    for idx in range(n_strokes):
        bbox = bboxes[idx]
        if _paint_stroke(grid, bbox, side_mask, BLUE) != len(_stroke_cells(bbox, side_mask)):
            return None

    components = _components_mask(grid == BLUE)
    if len(components) != n_strokes:
        return None

    counts = list(range(1, n_strokes + 1))
    random.shuffle(counts)
    for idx, count in enumerate(counts):
        choices = _interior_cells(bboxes[idx])
        if len(choices) < count:
            return None
        random.shuffle(choices)
        for r, c in choices[:count]:
            grid[r, c] = RED

    return _random_symmetry(grid)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_canonical_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        if not np.any(out == BLUE) or not np.any(out == GREY):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
