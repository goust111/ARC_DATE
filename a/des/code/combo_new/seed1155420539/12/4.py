from __future__ import annotations

from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
AZURE = 8
ANCHOR_COLORS = (1, 2, 3, 5, 6, 7, 9)


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            boxes.append((min(rows), min(cols), max(rows), max(cols)))
    return boxes


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        return [(r1, c) for c in range(lo, hi + 1)]
    if c1 == c2:
        lo, hi = sorted((r1, r2))
        return [(r, c1) for r in range(lo, hi + 1)]
    raise AssertionError("anchor pairs must be axis-aligned")


def _anchor_is_on_azure(pos: tuple[int, int], azure_boxes: Iterable[tuple[int, int, int, int]]) -> bool:
    r, c = pos
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in azure_boxes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each aligned anchor pair with a solid or dashed line determined by whether the two anchors sit on azure rectangles or on black background.

    Input structure:
    - The background is black.
    - Several non-overlapping solid azure rectangles are scattered on the grid.
    - Every other visible color appears exactly twice as two single-pixel anchors.
    - Each same-colored pair is aligned either horizontally or vertically, and each anchor lies either on black or inside an azure rectangle.

    Rule:
    - For each color pair, draw the straight axis-aligned connection between the two anchors.
    - If both anchors sit on azure, fill the whole interior of the segment with that anchor color.
    - If both anchors sit on black, draw a dashed line in that anchor color, coloring every other cell starting from the anchor closer to the grid origin.
    - If one anchor sits on azure and the other on black, draw a solid yellow line.
    - Keep the two endpoint anchor pixels in their original color, and leave all untouched azure cells unchanged.

    Key insight:
    - The anchor color identifies which two pixels to connect, but the visible azure rectangles encode a second layer of information: each endpoint's support type determines whether the connection is solid, dashed, or recolored to yellow.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h <= 30 and w <= 30

    out = g.copy()
    azure_boxes = _component_bboxes(g == AZURE)
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, AZURE)]

    for color in colors:
        pts = np.argwhere(g == color)
        assert len(pts) == 2, f"color {color} must appear exactly twice"
        a = (int(pts[0, 0]), int(pts[0, 1]))
        b = (int(pts[1, 0]), int(pts[1, 1]))
        cells = _segment_cells(a, b)
        start, end = sorted((a, b))
        a_on_azure = _anchor_is_on_azure(a, azure_boxes)
        b_on_azure = _anchor_is_on_azure(b, azure_boxes)

        if a_on_azure and b_on_azure:
            paint_color = color
            for rr, cc in cells[1:-1]:
                out[rr, cc] = paint_color
        elif (not a_on_azure) and (not b_on_azure):
            assert start == cells[0] and end == cells[-1]
            for offset, (rr, cc) in enumerate(cells[1:-1], start=1):
                if offset % 2 == 0:
                    out[rr, cc] = color
        else:
            for rr, cc in cells[1:-1]:
                out[rr, cc] = YELLOW

    return out


# --------------- generator ---------------

GRID_MIN = 16
GRID_MAX = 30
PAIR_COUNT_RANGE = (3, 5)
RECT_SIZE_RANGE = (3, 7)
EXTRA_RECT_RANGE = (0, 2)
PAIR_RETRIES = 200
RECT_RETRIES = 150
GEN_RETRIES = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _bbox_has_margin(
    bbox: tuple[int, int, int, int],
    others: Iterable[tuple[int, int, int, int]],
    margin: int = 1,
) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in others:
        if not (r1 + margin < or0 or or1 + margin < r0 or c1 + margin < oc0 or oc1 + margin < c0):
            return False
    return True


def _candidate_rectangles_for_anchor(
    anchor: tuple[int, int],
    shape: tuple[int, int],
    occupied: Iterable[tuple[int, int, int, int]],
    all_anchors: set[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    r, c = anchor
    candidates: list[tuple[int, int, int, int]] = []
    for rect_h in range(RECT_SIZE_RANGE[0], min(RECT_SIZE_RANGE[1], h) + 1):
        for rect_w in range(RECT_SIZE_RANGE[0], min(RECT_SIZE_RANGE[1], w) + 1):
            for top in range(max(0, r - rect_h + 2), min(r, h - rect_h) + 1):
                for left in range(max(0, c - rect_w + 2), min(c, w - rect_w) + 1):
                    bottom = top + rect_h - 1
                    right = left + rect_w - 1
                    if not (top < r < bottom and left < c < right):
                        continue
                    bbox = (top, left, bottom, right)
                    if not _bbox_has_margin(bbox, occupied, margin=1):
                        continue
                    bad_anchor = False
                    for other in all_anchors:
                        if other == anchor:
                            continue
                        orow, ocol = other
                        if top <= orow <= bottom and left <= ocol <= right:
                            bad_anchor = True
                            break
                    if not bad_anchor:
                        candidates.append(bbox)
    return candidates


def _place_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    used_paths: set[tuple[int, int]],
    existing_anchors: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    for _ in range(PAIR_RETRIES):
        horizontal = bool(rng.integers(0, 2))
        if horizontal:
            if w < 8:
                continue
            row = _randint(rng, 2, h - 3)
            c1 = _randint(rng, 2, w - 6)
            c2 = _randint(rng, c1 + 3, w - 3)
            a = (row, c1)
            b = (row, c2)
        else:
            if h < 8:
                continue
            col = _randint(rng, 2, w - 3)
            r1 = _randint(rng, 2, h - 6)
            r2 = _randint(rng, r1 + 3, h - 3)
            a = (r1, col)
            b = (r2, col)

        if any(_chebyshev(a, p) <= 2 or _chebyshev(b, p) <= 2 for p in existing_anchors):
            continue
        cells = _segment_cells(a, b)
        if any(cell in used_paths for cell in cells):
            continue
        used_paths.update(cells)
        return a, b
    return None


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        color = int(spec["color"])
        a, b = spec["points"]  # type: ignore[misc]
        a_on_azure, b_on_azure = spec["azure_flags"]  # type: ignore[misc]
        cells = _segment_cells(a, b)
        if a_on_azure and b_on_azure:
            for rr, cc in cells[1:-1]:
                out[rr, cc] = color
        elif (not a_on_azure) and (not b_on_azure):
            for offset, (rr, cc) in enumerate(cells[1:-1], start=1):
                if offset % 2 == 0:
                    out[rr, cc] = color
        else:
            for rr, cc in cells[1:-1]:
                out[rr, cc] = YELLOW
    return out


def _add_extra_rectangles(
    rng: np.random.Generator,
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    anchors: set[tuple[int, int]],
) -> None:
    h, w = shape
    extra_target = _randint(rng, EXTRA_RECT_RANGE[0], EXTRA_RECT_RANGE[1])
    for _ in range(extra_target):
        placed = False
        for _ in range(RECT_RETRIES):
            rect_h = _randint(rng, RECT_SIZE_RANGE[0], min(RECT_SIZE_RANGE[1], h))
            rect_w = _randint(rng, RECT_SIZE_RANGE[0], min(RECT_SIZE_RANGE[1], w))
            top = _randint(rng, 0, h - rect_h)
            left = _randint(rng, 0, w - rect_w)
            bbox = (top, left, top + rect_h - 1, left + rect_w - 1)
            if not _bbox_has_margin(bbox, rects, margin=1):
                continue
            if any(top <= r <= bbox[2] and left <= c <= bbox[3] for r, c in anchors):
                continue
            rects.append(bbox)
            placed = True
            break
        if not placed:
            return


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, GRID_MIN, GRID_MAX)
    w = _randint(rng, GRID_MIN, GRID_MAX)
    n_pairs = _randint(rng, PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    if n_pairs > len(ANCHOR_COLORS):
        return None

    branch_types = ["aa", "bb", "ab"]
    while len(branch_types) < n_pairs:
        branch_types.append(("aa", "bb", "ab")[int(rng.integers(0, 3))])
    rng.shuffle(branch_types)

    colors = list(rng.choice(np.array(ANCHOR_COLORS, dtype=int), size=n_pairs, replace=False))

    used_paths: set[tuple[int, int]] = set()
    anchors: list[tuple[int, int]] = []
    specs: list[dict[str, object]] = []

    for branch, color in zip(branch_types, colors):
        placed = _place_pair(rng, (h, w), used_paths, anchors)
        if placed is None:
            return None
        a, b = placed
        if branch == "aa":
            azure_flags = (True, True)
        elif branch == "bb":
            azure_flags = (False, False)
        else:
            azure_flags = (True, False) if bool(rng.integers(0, 2)) else (False, True)
        anchors.extend([a, b])
        specs.append({"color": int(color), "points": (a, b), "azure_flags": azure_flags})

    anchor_set = set(anchors)
    azure_anchor_positions = [
        point
        for spec in specs
        for point, on_azure in zip(spec["points"], spec["azure_flags"])  # type: ignore[arg-type]
        if on_azure
    ]
    rng.shuffle(azure_anchor_positions)

    rects: list[tuple[int, int, int, int]] = []
    for anchor in azure_anchor_positions:
        candidates = _candidate_rectangles_for_anchor(anchor, (h, w), rects, anchor_set)
        if not candidates:
            return None
        rects.append(candidates[int(rng.integers(0, len(candidates)))])

    _add_extra_rectangles(rng, (h, w), rects, anchor_set)

    grid = np.zeros((h, w), dtype=int)
    for bbox in rects:
        _paint_rect(grid, bbox, AZURE)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["points"]:  # type: ignore[misc]
            grid[r, c] = color

    expected = _expected_output(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if len({branch for branch in branch_types}) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
