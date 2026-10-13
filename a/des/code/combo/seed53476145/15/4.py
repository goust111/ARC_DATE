from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
PALETTE = tuple(color for color in range(10) if color != AZURE)
EDGES = ("top", "bottom", "left", "right")

GRID_SIZE_RANGE = (10, 18)
OBJECT_COUNT_CHOICES = (2, 3, 4)
RECT_HEIGHT_RANGE = (2, 5)
RECT_WIDTH_RANGE = (2, 5)
NEAR_EDGE_MAX = 3
PLACE_RETRIES = 80
GENERATE_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )

    return components


def _edge_distances(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return {
        "top": int(r0),
        "bottom": int(h - 1 - r1),
        "left": int(c0),
        "right": int(w - 1 - c1),
    }


def _nearest_edge(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> str:
    distances = _edge_distances(bbox, shape)
    assert min(distances.values()) > 0, "objects must stay off the border"
    best = min(distances.values())
    nearest = [edge for edge, dist in distances.items() if dist == best]
    assert len(set(distances.values())) == 4, "expected four distinct border distances"
    assert len(nearest) == 1, "expected a unique nearest border"
    return nearest[0]


def _nearest_edge_if_valid(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> str | None:
    distances = _edge_distances(bbox, shape)
    if min(distances.values()) <= 0:
        return None
    if len(set(distances.values())) != 4:
        return None
    best = min(distances.values())
    nearest = [edge for edge, dist in distances.items() if dist == best]
    if len(nearest) != 1:
        return None
    return nearest[0]


def _compress_rectangles(
    shape: tuple[int, int],
    rectangles: list[tuple[int, tuple[int, int, int, int]]],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color, bbox in rectangles:
        r0, c0, r1, c1 = bbox
        edge = _nearest_edge(bbox, shape)
        if edge == "top":
            out[r0, c0 : c1 + 1] = int(color)
        elif edge == "bottom":
            out[r1, c0 : c1 + 1] = int(color)
        elif edge == "left":
            out[r0 : r1 + 1, c0] = int(color)
        else:
            assert edge == "right"
            out[r0 : r1 + 1, c1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each solid rectangle toward its unique nearest border.

    Every non-background object is a separated 4-connected solid rectangle.
    A rectangle whose bbox is nearest to the top or bottom border collapses
    into a single row on that side of the bbox. A rectangle whose bbox is
    nearest to the left or right border collapses into a single column on
    that side of the bbox. Colors stay unchanged and all other cells become
    background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground rectangle"

    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in components:
        color = int(comp["color"])
        coords = comp["coords"]
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(coords) == area, "each component must be a solid rectangle"
        rectangles.append((color, bbox))

    return _compress_rectangles(g.shape, rectangles, bg)


def _inflate_rect(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _sample_rect_for_edge(
    shape: tuple[int, int], edge: str
) -> tuple[int, int, int, int] | None:
    h, w = shape
    max_h = min(RECT_HEIGHT_RANGE[1], h - 2)
    max_w = min(RECT_WIDTH_RANGE[1], w - 2)
    if max_h < RECT_HEIGHT_RANGE[0] or max_w < RECT_WIDTH_RANGE[0]:
        return None

    rect_h = random.randint(RECT_HEIGHT_RANGE[0], max_h)
    rect_w = random.randint(RECT_WIDTH_RANGE[0], max_w)

    if edge in {"top", "bottom"}:
        near_limit = min(NEAR_EDGE_MAX, h - rect_h - 1)
        if near_limit < 1:
            return None
        near = random.randint(1, near_limit)
        r0 = near if edge == "top" else h - rect_h - near
        c0 = random.randint(1, w - rect_w - 1)
    else:
        near_limit = min(NEAR_EDGE_MAX, w - rect_w - 1)
        if near_limit < 1:
            return None
        near = random.randint(1, near_limit)
        c0 = near if edge == "left" else w - rect_w - near
        r0 = random.randint(1, h - rect_h - 1)

    bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    if _nearest_edge_if_valid(bbox, shape) != edge:
        return None
    return bbox


def _place_rectangles(
    shape: tuple[int, int], target_edges: list[str]
) -> list[tuple[int, tuple[int, int, int, int]]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample(PALETTE, len(target_edges))
    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []

    for color, edge in zip(colors, target_edges):
        placed = False
        for _ in range(PLACE_RETRIES):
            bbox = _sample_rect_for_edge(shape, edge)
            if bbox is None:
                continue
            rr0, cc0, rr1, cc1 = _inflate_rect(bbox, shape, pad=1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            rectangles.append((color, bbox))
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return None

    return rectangles


def _draw_rectangles(
    shape: tuple[int, int], rectangles: list[tuple[int, tuple[int, int, int, int]]]
) -> np.ndarray:
    grid = np.full(shape, AZURE, dtype=int)
    for color, (r0, c0, r1, c1) in rectangles:
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _generate_candidate() -> np.ndarray | None:
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    min_side = GRID_SIZE_RANGE[0] + max(0, object_count - 2) * 2
    if min_side > GRID_SIZE_RANGE[1]:
        return None

    shape = (
        random.randint(min_side, GRID_SIZE_RANGE[1]),
        random.randint(min_side, GRID_SIZE_RANGE[1]),
    )
    target_edges = random.sample(list(EDGES), object_count)
    rectangles = _place_rectangles(shape, target_edges)
    if rectangles is None:
        return None

    grid = _draw_rectangles(shape, rectangles)
    out = main(grid)
    expected = _compress_rectangles(shape, rectangles, AZURE)

    if np.array_equal(grid, out):
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
