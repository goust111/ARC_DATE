from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= {int(c) for c in exclude}
    return sorted(colors)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[Coord] = deque([(r, c)])
        seen[r, c] = True
        coords: list[Coord] = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(coords)
    return [np.asarray(comp, dtype=int) for comp in comps]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_mask(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    return _bbox_coords(coords)


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _peel_once(mask: np.ndarray) -> np.ndarray:
    """Remove every foreground cell touching background in the 4-neighborhood."""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < H and 0 <= nc < W) or not mask[nr, nc]:
                out[r, c] = False
                break
    return out


def _peel_sequence(mask: np.ndarray) -> list[np.ndarray]:
    seq: list[np.ndarray] = []
    cur = np.asarray(mask, dtype=bool).copy()
    while True:
        cur = _peel_once(cur)
        if not cur.any():
            return seq
        seq.append(cur.copy())


def _extract_shape(grid: np.ndarray) -> tuple[int, BBox, np.ndarray]:
    bg = _get_bg(grid)
    fg_colors = _colors_present(grid, exclude={bg})
    assert len(fg_colors) == 1, "expected one non-background shape color"
    color = int(fg_colors[0])

    mask = grid == color
    comps = _components_mask(mask)
    assert len(comps) == 1, "expected one connected foreground shape"
    bbox = _bbox_coords(comps[0])
    r0, c0, r1, c1 = bbox
    crop = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert crop.any(), "shape crop must be non-empty"
    return color, bbox, crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the left solid shape rightward as a decaying echo sequence.

    Input structure:
    - The background is the dominant color.
    - The only foreground object is a single-color 4-connected solid shape in
      the left part of the grid.
    - The space to the right of the shape is background, leaving room for the echoes.

    Rule:
    - Let the horizontal period be the original shape bbox width plus one.
    - Keep the original shape, then for echo k place the result of peeling the
      original shape k times one period farther to the right.
    - A peel removes every current shape pixel that is 4-adjacent to background;
      after each peel, crop the remaining shape to its own bbox and align that
      bbox's upper-left corner to the period anchor. Stop when peeling is empty.

    Key insight:
    The copies are not translations or resizes; consecutive echoes are exactly
    the successive 4-neighbor boundary erosions of the original solid shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    H, W = g.shape
    bg = _get_bg(g)
    color, bbox, shape = _extract_shape(g)
    r0, c0, _r1, c1 = bbox
    period = c1 - c0 + 2

    out = g.copy()
    for k, peeled in enumerate(_peel_sequence(shape), start=1):
        echo = _crop_to_bbox(peeled)
        eh, ew = echo.shape
        rr = r0
        cc = c0 + k * period
        assert rr + eh <= H and cc + ew <= W, "echo sequence must fit in the grid"
        region = out[rr : rr + eh, cc : cc + ew]
        assert np.all(region[echo] == bg), "echoes must land on background cells"
        region[echo] = color

    return out


# --------------- generator ---------------

ALL_COLORS = tuple(range(10))
SHAPE_HEIGHT_RANGE = (6, 12)
SHAPE_WIDTH_RANGE = (5, 8)
TOP_MARGIN_MAX = 4
LEFT_MARGIN_MAX = 2
RIGHT_MARGIN_MAX = 3
BOTTOM_MARGIN_MAX = 4
MIN_ECHO_COUNT = 2
MAX_GENERATE_ATTEMPTS = 600
SHAPE_ATTEMPTS = 120


def _is_full_rectangle(mask: np.ndarray) -> bool:
    return bool(mask.size and np.all(mask))


def _random_walk_margins(length: int, cap: int) -> list[int]:
    values = [random.randint(0, cap)]
    for _ in range(1, length):
        step = random.choice((-1, 0, 0, 1))
        values.append(max(0, min(cap, values[-1] + step)))
    return values


def _random_shape_mask() -> np.ndarray | None:
    h = random.randint(*SHAPE_HEIGHT_RANGE)
    w = random.randint(*SHAPE_WIDTH_RANGE)
    center = random.randint(2, w - 3)
    left_cap = center
    right_cap = w - 1 - center

    left = _random_walk_margins(h, left_cap)
    right = _random_walk_margins(h, right_cap)

    # Add one or two side notches to make non-rectangular erosions common.
    mid = random.randint(1, h - 2)
    for _ in range(random.randint(1, 2)):
        side = random.choice(("left", "right"))
        start = random.randint(1, h - 3)
        end = random.randint(start, min(h - 2, start + random.randint(1, 3)))
        depth = random.randint(1, left_cap if side == "left" else right_cap)
        for r in range(start, end + 1):
            if side == "left":
                left[r] = max(left[r], depth)
            else:
                right[r] = max(right[r], depth)

    # Force a full-width row after carving so the period clue is visible from
    # the same bbox width used by the solver.
    left[mid] = 0
    right[mid] = 0

    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        lo = min(left[r], center)
        hi = max(center, w - 1 - right[r])
        if lo > hi:
            return None
        mask[r, lo : hi + 1] = True

    if _is_full_rectangle(mask):
        return None
    if len(_components_mask(mask)) != 1:
        return None
    return mask


def _choose_shape() -> tuple[np.ndarray, list[np.ndarray]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        mask = _random_shape_mask()
        if mask is None:
            continue
        echoes = _peel_sequence(mask)
        if len(echoes) < MIN_ECHO_COUNT:
            continue
        echo_crops = [_crop_to_bbox(e) for e in echoes]
        areas = [int(mask.sum()), *[int(e.sum()) for e in echoes]]
        if any(a <= b for a, b in zip(areas, areas[1:])):
            continue
        if _is_full_rectangle(echo_crops[0]):
            continue
        return mask, echo_crops
    return None


def _render_expected(inp: np.ndarray, color: int, bbox: BBox, shape: np.ndarray) -> np.ndarray:
    out = inp.copy()
    r0, c0, _r1, c1 = bbox
    period = c1 - c0 + 2
    for k, peeled in enumerate(_peel_sequence(shape), start=1):
        echo = _crop_to_bbox(peeled)
        rr = r0
        cc = c0 + k * period
        out[rr : rr + echo.shape[0], cc : cc + echo.shape[1]][echo] = color
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    color: int,
    bbox: BBox,
    shape: np.ndarray,
    echo_crops: list[np.ndarray],
) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    r0, c0, r1, c1 = bbox
    period = c1 - c0 + 2
    if len(_components_mask(inp == color)) != 1:
        return False
    if np.any(inp[:, c1 + 1 :] != bg):
        return False
    if not np.array_equal(expected[r0 : r1 + 1, c0 : c1 + 1] == color, shape):
        return False
    if len(echo_crops) < MIN_ECHO_COUNT:
        return False

    previous_area = int(shape.sum())
    for k, echo in enumerate(echo_crops, start=1):
        area = int(echo.sum())
        if not (0 < area < previous_area):
            return False
        rr = r0
        cc = c0 + k * period
        if cc <= c1:
            return False
        view = expected[rr : rr + echo.shape[0], cc : cc + echo.shape[1]]
        if not np.array_equal(view == color, echo):
            return False
        sep_col = c0 + k * period - 1
        if not np.all(expected[r0 : r1 + 1, sep_col] == bg):
            return False
        previous_area = area

    # Reject samples that could be explained as just translating the original
    # or as a rectangular shrink in the first visible echo.
    if echo_crops[0].shape == shape.shape and np.array_equal(echo_crops[0], shape):
        return False
    if _is_full_rectangle(echo_crops[0]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    chosen = _choose_shape()
    if chosen is None:
        return None
    shape, echo_crops = chosen
    h, w = shape.shape
    period = w + 1

    r0 = random.randint(0, TOP_MARGIN_MAX)
    c0 = random.randint(0, LEFT_MARGIN_MAX)
    max_echo_right = max(
        c0 + k * period + echo.shape[1]
        for k, echo in enumerate(echo_crops, start=1)
    )
    min_W = max(c0 + w, max_echo_right)
    if min_W > 30:
        return None
    W = min_W + random.randint(0, min(RIGHT_MARGIN_MAX, 30 - min_W))

    min_H = r0 + h
    if min_H > 30:
        return None
    H = min_H + random.randint(0, min(BOTTOM_MARGIN_MAX, 30 - min_H))

    bg = random.choice(ALL_COLORS)
    color = random.choice([c for c in ALL_COLORS if c != bg])
    inp = np.full((H, W), bg, dtype=int)
    inp[r0 : r0 + h, c0 : c0 + w][shape] = color

    if _get_bg(inp) != bg:
        return None

    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
    expected = _render_expected(inp, color, bbox, shape)
    if not _semantic_checks(inp, expected, bg, color, bbox, shape, echo_crops):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid decaying-echo input")
