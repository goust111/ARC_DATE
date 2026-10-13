from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
GREY = 5

GRID_ROWS_RANGE = (14, 18)
GRID_COLS_RANGE = (14, 18)
RECT_COUNT_CHOICES = (2, 3)
RECT_SIZE_CHOICES = tuple((h, w) for h in range(3, 7) for w in range(3, 7))
MAX_GENERATE_ATTEMPTS = 800
MAX_RECT_PLACE_ATTEMPTS = 300


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    """Return 4-connected components for one color as coord arrays."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expand_bbox(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    cell_set = {tuple(map(int, rc)) for rc in coords}
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return cell_set == border


def _center_from_bbox(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _template_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([[r0, c0]])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0))


def _extract_objects(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    red_comps = _components_of_color(grid, RED)
    assert len(red_comps) == 1, "expected exactly one red template"
    template_coords = red_comps[0]
    assert 3 <= len(template_coords) <= 5, "expected a small red template"
    template_mask, _origin = _template_mask(template_coords)
    assert template_mask.shape[0] <= 3 and template_mask.shape[1] <= 3

    grey_comps = _components_of_color(grid, GREY)
    assert 2 <= len(grey_comps) <= 3, "expected 2-3 grey hollow rectangles"
    rects = []
    for comp in grey_comps:
        assert _is_hollow_rectangle(comp), "grey objects must be hollow rectangles"
        rects.append(_bbox(comp))

    rows = [_center_from_bbox(box)[0] for box in rects]
    cols = [_center_from_bbox(box)[1] for box in rects]
    assert len(set(rows)) == len(rows), "rectangle center rows must be distinct"
    assert len(set(cols)) == len(cols), "rectangle center cols must be distinct"
    return template_mask, rects


def _render_output(
    shape: tuple[int, int], template_mask: np.ndarray, rects: list[tuple[int, int, int, int]]
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    centers = [_center_from_bbox(box) for box in rects]
    points = {(centers[i][0], centers[j][1]) for i in range(len(centers)) for j in range(len(centers)) if i != j}

    th, tw = template_mask.shape
    for r0, c0 in sorted(points):
        assert 0 <= r0 and 0 <= c0 and r0 + th <= h and c0 + tw <= w
        patch = out[r0 : r0 + th, c0 : c0 + tw]
        patch[template_mask] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp red template copies at emergent intersections of hollow-rectangle centerlines.

    Input structure:
    - The grid has a black background.
    - Two or three grey hollow rectangles appear as separated 1-pixel-thick outlines.
    - One small red 4-connected template sits in a corner and does not touch the rectangles.

    Rule:
    - Compute each rectangle's center row and center column from its bounding box.
    - Treat every center row and center column as a full-length hidden crosshair.
    - For every ordered pair of distinct rectangles A and B, place one copy of the red template
      with its bbox upper-left corner at (center_row(A), center_col(B)).
    - Remove the original template and all grey rectangle pixels; keep only the stamped copies on black.

    Key insight:
    - The stamp locations are not marked explicitly in the input; they emerge only after combining
      center rows from one rectangle with center columns from another.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, GREY})

    template_mask, rects = _extract_objects(g)
    return _render_output(g.shape, template_mask, rects)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q: deque[tuple[int, int]] = deque([tuple(map(int, coords[0]))])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_template_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    for h in (2, 3):
        for w in (2, 3):
            for bits in range(1 << (h * w)):
                mask = np.zeros((h, w), dtype=bool)
                for i in range(h * w):
                    if bits & (1 << i):
                        mask[i // w, i % w] = True
                area = int(mask.sum())
                if not (3 <= area <= 5):
                    continue
                if not _mask_connected(mask):
                    continue
                if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                shapes.append(mask.copy())
    # Deduplicate exact masks while preserving variety.
    unique: dict[bytes, np.ndarray] = {}
    for mask in shapes:
        unique.setdefault(mask.tobytes() + bytes(mask.shape), mask)
    return list(unique.values())


TEMPLATE_LIBRARY = _build_template_library()


def _draw_hollow_rectangle(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _stamp_from_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _corner_positions(shape: tuple[int, int], template_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    th, tw = template_shape
    return [(0, 0), (0, w - tw), (h - th, 0), (h - th, w - tw)]


def _rect_center_from_top_left(top: int, left: int, h: int, w: int) -> tuple[int, int]:
    return top + (h - 1) // 2, left + (w - 1) // 2


def _boxes_with_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not _rects_overlap(_expand_bbox(a, gap, (10**6, 10**6)), b)


def _stamp_boxes_non_overlapping(
    points: list[tuple[int, int]], template_shape: tuple[int, int], grid_shape: tuple[int, int]
) -> bool:
    th, tw = template_shape
    h, w = grid_shape
    boxes = []
    for r0, c0 in points:
        if not (0 <= r0 and 0 <= c0 and r0 + th <= h and c0 + tw <= w):
            return False
        boxes.append((r0, c0, r0 + th - 1, c0 + tw - 1))
    for a, b in combinations(boxes, 2):
        if _rects_overlap(a, b):
            return False
    return True


def _template_not_touching_rects(
    template_box: tuple[int, int, int, int], rects: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> bool:
    padded_template = _expand_bbox(template_box, 1, shape)
    return all(not _rects_overlap(padded_template, box) for box in rects)


def _pick_rectangle_dims(count: int) -> list[tuple[int, int]]:
    dims_pool = list(RECT_SIZE_CHOICES)
    random.shuffle(dims_pool)
    dims = dims_pool[:count]
    if not any(h % 2 == 0 or w % 2 == 0 for h, w in dims):
        even_dims = [(h, w) for h, w in RECT_SIZE_CHOICES if h % 2 == 0 or w % 2 == 0]
        dims[0] = random.choice(even_dims)
        if len(set(dims)) < len(dims):
            return _pick_rectangle_dims(count)
    return dims


def _place_rectangles(
    shape: tuple[int, int], template_box: tuple[int, int, int, int], template_shape: tuple[int, int], count: int
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    th, tw = template_shape
    dims = _pick_rectangle_dims(count)
    rects: list[tuple[int, int, int, int]] = []

    for rh, rw in dims:
        candidates = []
        max_top = h - rh
        max_left = w - rw
        for top in range(max_top + 1):
            for left in range(max_left + 1):
                center_r, center_c = _rect_center_from_top_left(top, left, rh, rw)
                if center_r + th > h or center_c + tw > w:
                    continue
                box = (top, left, top + rh - 1, left + rw - 1)
                if not _template_not_touching_rects(template_box, [box], shape):
                    continue
                ok = True
                for other in rects:
                    other_center_r, other_center_c = _center_from_bbox(other)
                    if abs(center_r - other_center_r) <= th or abs(center_c - other_center_c) <= tw:
                        ok = False
                        break
                    if _rects_overlap(_expand_bbox(box, 1, shape), other):
                        ok = False
                        break
                if ok:
                    candidates.append(box)
        if not candidates:
            return None
        rects.append(random.choice(candidates))

    centers = [_center_from_bbox(box) for box in rects]
    points = [(centers[i][0], centers[j][1]) for i in range(len(centers)) for j in range(len(centers)) if i != j]
    if not _stamp_boxes_non_overlapping(points, template_shape, shape):
        return None
    stamp_boxes = [(r, c, r + th - 1, c + tw - 1) for r, c in points]
    for box in stamp_boxes:
        if _rects_overlap(box, template_box):
            return None
    return rects


def _build_expected(shape: tuple[int, int], template_mask: np.ndarray, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    return _render_output(shape, template_mask, rects)


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    template_mask: np.ndarray,
    template_origin: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    red_comps = _components_of_color(grid, RED)
    if len(red_comps) != 1:
        return False
    mask, origin = _template_mask(red_comps[0])
    if origin != template_origin or not np.array_equal(mask, template_mask):
        return False

    grey_comps = _components_of_color(grid, GREY)
    if len(grey_comps) != len(rects):
        return False
    parsed_rects = sorted(_bbox(comp) for comp in grey_comps)
    if parsed_rects != sorted(rects):
        return False
    if any(not _is_hollow_rectangle(comp) for comp in grey_comps):
        return False

    centers = [_center_from_bbox(box) for box in rects]
    if len({r for r, _c in centers}) != len(centers):
        return False
    if len({c for _r, c in centers}) != len(centers):
        return False

    points = [(centers[i][0], centers[j][1]) for i in range(len(centers)) for j in range(len(centers)) if i != j]
    if len(points) != len(set(points)):
        return False
    if not _stamp_boxes_non_overlapping(points, template_mask.shape, grid.shape):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    shape = (h, w)
    template_mask = random.choice(TEMPLATE_LIBRARY)
    th, tw = template_mask.shape
    template_top, template_left = random.choice(_corner_positions(shape, template_mask.shape))
    template_box = (template_top, template_left, template_top + th - 1, template_left + tw - 1)
    rect_count = random.choice(RECT_COUNT_CHOICES)

    rects = _place_rectangles(shape, template_box, template_mask.shape, rect_count)
    if rects is None:
        return None

    expected = _build_expected(shape, template_mask, rects)
    grid = np.zeros(shape, dtype=int)
    for box in rects:
        _draw_hollow_rectangle(grid, box, GREY)
    _stamp_from_mask(grid, template_mask, template_top, template_left, RED)

    if not _validate_instance(grid, expected, template_mask, (template_top, template_left), rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
