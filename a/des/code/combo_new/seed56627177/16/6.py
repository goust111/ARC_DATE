from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
TRACK_COLORS = (1, 2, 5, 6, 7, 8, 9)
GEN_RETRIES = 500
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _clockwise_perimeter(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        pts.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def _analyze_track(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "track must enclose a non-empty interior"

    perimeter = _clockwise_perimeter(r0, c0, r1, c1)
    coord_set = set(coords)
    assert len(coord_set) == len(perimeter), "track component must match a rectangle perimeter"
    assert all(pt in coord_set for pt in perimeter), "track component must fill the entire rectangle perimeter"

    perimeter_colors = [int(grid[r, c]) for r, c in perimeter]
    base_colors = {color for color in perimeter_colors if color != YELLOW}
    assert len(base_colors) == 1, "each ring must have exactly one base track color"
    track_color = next(iter(base_colors))
    assert track_color not in (BLACK, GREEN, YELLOW)

    carriages = [i for i, (r, c) in enumerate(perimeter) if int(grid[r, c]) == YELLOW]
    step = int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == GREEN))

    return {
        "bbox": (r0, c0, r1, c1),
        "perimeter": perimeter,
        "track_color": track_color,
        "carriages": carriages,
        "step": step,
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expand_bbox(
    bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each yellow carriage clockwise around its rectangular ring by the number of green interior markers.

    Input structure:
    - The background is black.
    - Each foreground object is a one-pixel-thick rectangular ring whose base color is consistent except for some yellow pixels placed on the ring.
    - The yellow ring pixels are the carriages, and the black interior of each ring contains several green marker pixels.

    Rule:
    - Process each rectangular ring independently.
    - Count the number of green pixels strictly inside that ring; call it N.
    - Move every yellow carriage N steps clockwise along the ring perimeter, with top edge left->right, right edge top->bottom, bottom edge right->left, and left edge bottom->top.
    - Cells vacated by carriages revert to the ring's base color, and the green interior markers stay unchanged.

    Key insight:
    - The green pixels do not move; their count is the encoded rotation distance for all yellow pixels on the surrounding loop.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    track_mask = (g != BLACK) & (g != GREEN)

    for coords in _find_components(track_mask):
        info = _analyze_track(g, coords)
        perimeter = info["perimeter"]
        ring_len = len(perimeter)
        step = info["step"] % ring_len if ring_len else 0

        for r, c in perimeter:
            out[r, c] = info["track_color"]

        for idx in info["carriages"]:
            rr, cc = perimeter[(idx + step) % ring_len]
            out[rr, cc] = YELLOW

    return out


def _place_track(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    track_color: int,
    carriage_count: int,
    marker_count: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    perimeter = _clockwise_perimeter(r0, c0, r1, c1)
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    assert len(interior) >= marker_count

    carriages = set(rng.sample(perimeter, carriage_count))
    markers = set(rng.sample(interior, marker_count))

    for r, c in perimeter:
        grid[r, c] = YELLOW if (r, c) in carriages else track_color
    for r, c in markers:
        grid[r, c] = GREEN


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    track_count = rng.randint(1, min(4, len(TRACK_COLORS)))
    min_side = max(12, 8 + 4 * track_count)
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    grid = np.full((h, w), BLACK, dtype=int)

    colors = rng.sample(TRACK_COLORS, track_count)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for track_color in colors:
        marker_count = rng.randint(1, 6)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            inner_h = rng.randint(2, min(8, h - 4))
            inner_w = rng.randint(2, min(8, w - 4))
            if inner_h * inner_w < marker_count:
                continue

            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if h - outer_h - 1 < 1 or w - outer_w - 1 < 1:
                continue

            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            padded = _expand_bbox(bbox, 1, grid.shape)
            if any(_boxes_overlap(padded, _expand_bbox(other, 1, grid.shape)) for other in placed_boxes):
                continue

            perimeter_len = 2 * (outer_h + outer_w) - 4
            max_carriages = min(6, max(1, perimeter_len // 3))
            carriage_count = rng.randint(1, max_carriages)
            _place_track(grid, bbox, track_color, carriage_count, marker_count, rng)
            placed_boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(solved, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangular-ring sample")
