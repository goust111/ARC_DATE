from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 9)
SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400
COMPONENT_RETRIES = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _find_reference_frame(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], set[str], list[tuple[int, int]]]:
    mask = (grid == GREY) | (grid == AZURE)
    candidates: list[tuple[int, tuple[int, int, int, int], set[str], list[tuple[int, int]]]] = []

    for comp in _components_of_mask(mask):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        if np.count_nonzero(mask[r0 : r1 + 1, c0 : c1 + 1]) != int(border.sum()):
            continue
        if not np.all(np.isin(sub[border], (GREY, AZURE))):
            continue
        if not np.all(sub[~border] == BLACK):
            continue

        markers = [(r, c) for r, c in comp if int(grid[r, c]) == AZURE]
        if not 1 <= len(markers) <= 4:
            continue

        sides: set[str] = set()
        valid_markers = True
        for r, c in markers:
            if r == r0 and c0 < c < c1:
                side = "top"
            elif r == r1 and c0 < c < c1:
                side = "bottom"
            elif c == c0 and r0 < r < r1:
                side = "left"
            elif c == c1 and r0 < r < r1:
                side = "right"
            else:
                valid_markers = False
                break
            if side in sides:
                valid_markers = False
                break
            sides.add(side)
        if not valid_markers:
            continue

        candidates.append((h * w, (r0, c0, r1, c1), sides, markers))

    assert candidates, "expected one grey reference frame with azure markers"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, bbox, sides, markers = candidates[0]
    return bbox, sides, markers


def _find_solid_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    mask = (grid != BLACK) & (grid != GREY) & (grid != AZURE)
    rectangles: list[dict[str, object]] = []

    for comp in _components_of_mask(mask):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = colors.pop()
        if np.all(sub == color):
            rectangles.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "area": len(comp),
                }
            )

    assert 3 <= len(rectangles) <= 5, "expected 3-5 colored solid rectangles"
    rectangles.sort(key=lambda item: int(item["area"]), reverse=True)
    return rectangles


def _draw_outline(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the scattered solid rectangles by area, redraw them as nested outlines
    inside the grey reference frame, and cut the same centered gap on every
    nested layer according to the frame's azure edge markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    (fr0, fc0, fr1, fc1), gap_sides, markers = _find_reference_frame(g)
    rectangles = _find_solid_rectangles(g)

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    for r, c in markers:
        out[r, c] = BLACK

    for idx, rect in enumerate(rectangles):
        top = fr0 + 2 + 2 * idx
        left = fc0 + 2 + 2 * idx
        bottom = fr1 - 2 - 2 * idx
        right = fc1 - 2 - 2 * idx
        assert bottom - top >= 2 and right - left >= 2, "frame too small for all nested layers"

        _draw_outline(out, (top, left, bottom, right), int(rect["color"]))

        mid_r = (top + bottom) // 2
        mid_c = (left + right) // 2
        if "top" in gap_sides:
            out[top, mid_c] = BLACK
        if "bottom" in gap_sides:
            out[bottom, mid_c] = BLACK
        if "left" in gap_sides:
            out[mid_r, left] = BLACK
        if "right" in gap_sides:
            out[mid_r, right] = BLACK

    return out


def _odd_values(lo: int, hi: int) -> list[int]:
    return [v for v in range(lo, hi + 1) if v % 2 == 1]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    _draw_outline(grid, bbox, GREY)


def _reserve(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
        return False
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _place_object_rectangles(
    grid: np.ndarray,
    occupied: np.ndarray,
    count: int,
) -> bool:
    h, w = grid.shape
    colors = random.sample(list(OBJECT_COLORS), count)
    used_areas: set[int] = set()

    for color in colors:
        placed = False
        for _ in range(COMPONENT_RETRIES):
            rh = random.randint(2, 6)
            rw = random.randint(2, 7)
            area = rh * rw
            if area in used_areas:
                continue

            max_r0 = h - rh - 2
            max_c0 = w - rw - 2
            if max_r0 < 1 or max_c0 < 1:
                return False

            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if not _reserve(occupied, bbox, pad=1):
                continue

            grid[r0 : r0 + rh, c0 : c0 + rw] = color
            used_areas.add(area)
            placed = True
            break
        if not placed:
            return False

    return True


def _place_markers(grid: np.ndarray, bbox: tuple[int, int, int, int], sides: list[str]) -> None:
    r0, c0, r1, c1 = bbox
    for side in sides:
        if side == "top":
            col = random.randint(c0 + 1, c1 - 1)
            grid[r0, col] = AZURE
        elif side == "bottom":
            col = random.randint(c0 + 1, c1 - 1)
            grid[r1, col] = AZURE
        elif side == "left":
            row = random.randint(r0 + 1, r1 - 1)
            grid[row, c0] = AZURE
        else:
            row = random.randint(r0 + 1, r1 - 1)
            grid[row, c1] = AZURE


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    min_frame_dim = 4 * count + 3

    height = random.randint(min_frame_dim + 8, min_frame_dim + 16)
    width = random.randint(min_frame_dim + 8, min_frame_dim + 16)

    frame_h_choices = _odd_values(min_frame_dim, height - 6)
    frame_w_choices = _odd_values(min_frame_dim, width - 6)
    if not frame_h_choices or not frame_w_choices:
        return None

    frame_h = random.choice(frame_h_choices)
    frame_w = random.choice(frame_w_choices)
    max_r0 = height - frame_h - 2
    max_c0 = width - frame_w - 2
    if max_r0 < 1 or max_c0 < 1:
        return None

    fr0 = random.randint(1, max_r0)
    fc0 = random.randint(1, max_c0)
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    grid = np.zeros((height, width), dtype=int)
    _draw_frame(grid, frame_bbox)

    marker_count = random.randint(1, 3)
    marker_sides = random.sample(list(SIDES), marker_count)
    _place_markers(grid, frame_bbox, marker_sides)

    occupied = np.zeros((height, width), dtype=bool)
    if not _reserve(occupied, frame_bbox, pad=1):
        return None

    if not _place_object_rectangles(grid, occupied, count):
        return None

    # Sanity-check the generated example against the solver's assumptions.
    rectangles = _find_solid_rectangles(grid)
    assert len(rectangles) == count
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
