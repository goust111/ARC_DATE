from __future__ import annotations

import random

import numpy as np


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.asarray(coords, dtype=int))

    return comps


def _classify_segment(coords: np.ndarray) -> tuple[str, int]:
    """Classify a monochrome component as one horizontal or vertical segment."""
    assert len(coords) >= 2, "every fragment must be at least two cells long"

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if len(rows) == 1:
        ordered = np.sort(cols)
        assert np.array_equal(ordered, np.arange(int(ordered[0]), int(ordered[-1]) + 1))
        return "horizontal", int(rows[0])
    if len(cols) == 1:
        ordered = np.sort(rows)
        assert np.array_equal(ordered, np.arange(int(ordered[0]), int(ordered[-1]) + 1))
        return "vertical", int(cols[0])
    raise AssertionError("each component must be a straight segment")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each color's full solid rectangle from four scattered edge fragments.

    Input structure:
    - The background is black.
    - Each foreground color appears in exactly four disconnected straight segments.
    - For every color, two segments are horizontal fragments of one rectangle's top and bottom edges, and two are vertical fragments of that rectangle's left and right edges.

    Rule:
    - For each color, identify its four connected segments.
    - Use the rows of the two horizontal segments as the rectangle's top and bottom boundaries, and the columns of the two vertical segments as the left and right boundaries.
    - Fill the entire axis-aligned rectangle defined by those boundaries with that color.

    Key insight:
    - The fragments do not need to be extended endpoint-to-endpoint; their shared color and row/column positions already encode the rectangle's four sides.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        comps = _components(g == color)
        assert len(comps) == 4, "each color must form exactly four disconnected fragments"

        horizontal_rows: list[int] = []
        vertical_cols: list[int] = []
        for comp in comps:
            orient, axis_value = _classify_segment(comp)
            if orient == "horizontal":
                horizontal_rows.append(axis_value)
            else:
                vertical_cols.append(axis_value)

        assert len(horizontal_rows) == 2 and len(vertical_cols) == 2
        top, bottom = sorted(horizontal_rows)
        left, right = sorted(vertical_cols)
        assert top < bottom and left < right

        out[top : bottom + 1, left : right + 1] = color

    return out


# --------------- generator ---------------

BG = 0
PALETTE = list(range(1, 10))
GRID_MIN = 12
GRID_MAX = 30
MIN_RECT_SIDE = 4
SIDE_CAP_BY_COUNT = {1: 13, 2: 11, 3: 9, 4: 8}
OBJECT_GAP = 1
PLACE_TRIES = 120
MAX_GENERATION_ATTEMPTS = 400


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r1 + pad)
    cc1 = min(occupied.shape[1] - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _sample_rect(h: int, w: int, num_objects: int, occupied: np.ndarray) -> tuple[int, int, int, int] | None:
    side_cap = SIDE_CAP_BY_COUNT[num_objects]
    h_hi = min(side_cap, h)
    w_hi = min(side_cap, w)
    if h_hi < MIN_RECT_SIDE or w_hi < MIN_RECT_SIDE:
        return None

    for _ in range(PLACE_TRIES):
        rh = random.randint(MIN_RECT_SIDE, h_hi)
        rw = random.randint(MIN_RECT_SIDE, w_hi)
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        r1 = r0 + rh - 1
        c1 = c0 + rw - 1
        if not _can_place(occupied, r0, c0, r1, c1, OBJECT_GAP):
            continue
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        return (r0, c0, r1, c1)
    return None


def _sample_count(h: int, w: int) -> int:
    max_objects = min(4, max(1, (h // 5) * (w // 5)))
    return random.randint(1, max_objects)


def _sample_segment(lo: int, hi: int) -> tuple[int, int]:
    length = random.randint(2, hi - lo + 1)
    start = random.randint(lo, hi - length + 1)
    return start, start + length - 1


def _paint_fragments(inp: np.ndarray, color: int, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect

    top_c0, top_c1 = _sample_segment(c0 + 1, c1 - 1)
    bottom_c0, bottom_c1 = _sample_segment(c0 + 1, c1 - 1)
    left_r0, left_r1 = _sample_segment(r0 + 1, r1 - 1)
    right_r0, right_r1 = _sample_segment(r0 + 1, r1 - 1)

    inp[r0, top_c0 : top_c1 + 1] = color
    inp[r1, bottom_c0 : bottom_c1 + 1] = color
    inp[left_r0 : left_r1 + 1, c0] = color
    inp[right_r0 : right_r1 + 1, c1] = color


def _build_pair() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    num_objects = _sample_count(h, w)
    colors = random.sample(PALETTE, num_objects)

    occupied = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(num_objects):
        rect = _sample_rect(h, w, num_objects, occupied)
        if rect is None:
            return None
        rects.append(rect)

    inp = np.full((h, w), BG, dtype=int)
    out = inp.copy()
    for color, rect in zip(colors, rects, strict=True):
        _paint_fragments(inp, color, rect)
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _build_pair()
        if pair is None:
            continue
        inp, _out = pair
        return inp
    raise RuntimeError("failed to generate a valid rectangle-fragment input")
