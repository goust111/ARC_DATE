from __future__ import annotations

import random
from collections import deque

import numpy as np


class Color:
    BLUE = 1
    RED = 2
    YELLOW = 4
    PINK = 6
    TEAL = 8


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _rect_border_coords(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    coords: list[Coord] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    blue = grid == Color.BLUE
    rects: list[dict] = []
    for coords in _components_mask(blue):
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "blue object must be a hollow rectangle"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        assert np.all(patch[border] == Color.BLUE), "rectangle border must be fully blue"
        assert not np.any(patch[~border] == Color.BLUE), "rectangle interior must be blue-free"
        assert int(coords.shape[0]) == int(border.sum()), "blue component must match one hollow rectangle border"
        marker_count = int(np.count_nonzero(patch[1:-1, 1:-1] == Color.RED))
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "marker_count": marker_count,
            }
        )
    rects.sort(key=lambda rect: rect["bbox"])
    assert rects, "expected at least one blue hollow rectangle"
    return rects


def _render_with_center(grid: np.ndarray, center: Coord) -> np.ndarray:
    cr, cc = center
    out = np.array(grid, dtype=int, copy=True)
    border_mask = out == Color.BLUE
    marker_mask = out == Color.RED

    out[cr, :] = Color.PINK
    out[:, cc] = Color.PINK

    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if r == cr or c == cc or border_mask[r, c]:
                continue
            d = max(abs(r - cr), abs(c - cc))
            if d > 0 and d % 2 == 0 and out[r, c] == Color.TEAL:
                out[r, c] = Color.YELLOW

    out[border_mask] = Color.BLUE
    out[marker_mask] = Color.RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the rectangle with the most red markers, center a full pink cross on it, and fill even Chebyshev rings behind all original objects.

    Input structure:
    - The background is teal.
    - Several blue hollow rectangles appear, separated from one another.
    - Rectangle interiors may contain red singleton marker pixels on the teal background.

    Rule:
    - Count the red markers strictly inside each blue rectangle and choose the unique rectangle with the largest count.
    - Use that rectangle's bbox center, rounded down in each axis, as the intersection of a full-width pink row and a full-height pink column.
    - For every pixel that is neither on the pink cross nor on any blue rectangle border, fill teal background pixels whose Chebyshev distance from the chosen center is a positive even number with yellow.
    - Redraw all blue rectangle borders and all red markers on top of the cross and ring fill.

    Key insight:
    - The geometric center that drives both the cross and the ring pattern is not given directly; it is selected by comparing marker counts across multiple candidate rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rects = _find_rectangles(g)
    counts = [rect["marker_count"] for rect in rects]
    max_count = max(counts)
    winners = [rect for rect in rects if rect["marker_count"] == max_count]
    assert len(winners) == 1, "rectangle with most markers must be unique"
    return _render_with_center(g, winners[0]["center"])


GRID_SIDE_RANGE = (15, 20)
RECT_COUNT_RANGE = (3, 5)
RECT_SIDE_RANGE = (3, 6)
MAX_MARKERS = 4
GENERATION_ATTEMPTS = 600


def _sample_counts(n_rects: int) -> tuple[list[int], int]:
    winner_idx = random.randrange(n_rects)
    winner_count = random.randint(1, MAX_MARKERS)
    others = [random.randint(0, winner_count - 1) for _ in range(n_rects - 1)]
    if winner_count > 1 and others:
        others[random.randrange(len(others))] = winner_count - 1
    counts: list[int] = []
    other_i = 0
    for i in range(n_rects):
        if i == winner_idx:
            counts.append(winner_count)
        else:
            counts.append(others[other_i])
            other_i += 1
    return counts, winner_idx


def _sample_dimensions(counts: list[int]) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for count in counts:
        valid = [
            (h, w)
            for h in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1)
            for w in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1)
            if (h - 2) * (w - 2) >= count
        ]
        dims.append(random.choice(valid))
    return dims


def _boxes_conflict(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _place_rectangles(h: int, w: int, dims: list[tuple[int, int]]) -> list[BBox] | None:
    order = sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    boxes: list[BBox | None] = [None] * len(dims)
    placed: list[BBox] = []
    for idx in order:
        rh, rw = dims[idx]
        candidates = [(r, c) for r in range(h - rh + 1) for c in range(w - rw + 1)]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if all(not _boxes_conflict(box, other) for other in placed):
                chosen = box
                break
        if chosen is None:
            return None
        boxes[idx] = chosen
        placed.append(chosen)
    return [box for box in boxes if box is not None]


def _place_markers(grid: np.ndarray, box: BBox, count: int) -> None:
    if count == 0:
        return
    r0, c0, r1, c1 = box
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for r, c in random.sample(cells, count):
        grid[r, c] = Color.RED


def _render_input(h: int, w: int, boxes: list[BBox], counts: list[int]) -> np.ndarray:
    grid = np.full((h, w), Color.TEAL, dtype=int)
    for box in boxes:
        for r, c in _rect_border_coords(box):
            grid[r, c] = Color.BLUE
    for box, count in zip(boxes, counts, strict=True):
        _place_markers(grid, box, count)
    return grid


def _winner_center_from_boxes(boxes: list[BBox], winner_idx: int) -> Coord:
    r0, c0, r1, c1 = boxes[winner_idx]
    return (r0 + r1) // 2, (c0 + c1) // 2


def _scenario_checks(boxes: list[BBox], counts: list[int], winner_idx: int) -> bool:
    winner_box = boxes[winner_idx]
    winner_area = (winner_box[2] - winner_box[0] + 1) * (winner_box[3] - winner_box[1] + 1)
    winner_count = counts[winner_idx]

    conditions = []
    conditions.append(any(
        ((box[2] - box[0] + 1) * (box[3] - box[1] + 1)) > winner_area
        for i, box in enumerate(boxes)
        if i != winner_idx
    ))
    conditions.append(any(count == winner_count - 1 for i, count in enumerate(counts) if i != winner_idx))
    conditions.append((winner_box[2] - winner_box[0] + 1) % 2 == 0 or (winner_box[3] - winner_box[1] + 1) % 2 == 0)
    conditions.append(any(box[0] < winner_box[0] for i, box in enumerate(boxes) if i != winner_idx))
    conditions.append(any(box[1] < winner_box[1] for i, box in enumerate(boxes) if i != winner_idx))
    return sum(bool(cond) for cond in conditions) >= 3


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        rects = _find_rectangles(grid)
        counts = [rect["marker_count"] for rect in rects]
        max_count = max(counts)
        winners = [i for i, count in enumerate(counts) if count == max_count]
        if len(winners) != 1:
            return False
        winner_idx = winners[0]
        out = main(grid)
        expected = _render_with_center(grid, rects[winner_idx]["center"])
        if not np.array_equal(out, expected):
            return False
        if np.array_equal(out, grid):
            return False
        if int(np.count_nonzero(out == Color.YELLOW)) == 0:
            return False
        if int(np.count_nonzero(out == Color.PINK)) == 0:
            return False

        plain = np.array(grid, copy=True)
        cr, cc = rects[winner_idx]["center"]
        plain[cr, :] = Color.PINK
        plain[:, cc] = Color.PINK
        if np.array_equal(plain, out):
            return False

        for i, rect in enumerate(rects):
            if i == winner_idx:
                continue
            alt = _render_with_center(grid, rect["center"])
            if np.array_equal(alt, out):
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_rects = random.randint(*RECT_COUNT_RANGE)
    counts, winner_idx = _sample_counts(n_rects)
    dims = _sample_dimensions(counts)
    boxes = _place_rectangles(h, w, dims)
    if boxes is None:
        return None
    if not _scenario_checks(boxes, counts, winner_idx):
        return None

    grid = _render_input(h, w, boxes, counts)
    if _winner_center_from_boxes(boxes, winner_idx) != _find_rectangles(grid)[winner_idx]["center"]:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 011-1")
