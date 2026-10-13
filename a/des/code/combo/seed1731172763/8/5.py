from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

PAIR_GROUPS = ((RED, BLUE), (GREEN, YELLOW), (ORANGE, MAROON))
UNCHANGED_COLORS = (MAGENTA, CYAN)
NON_FRAME_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN, MAROON)

HALF_SIZE_RANGE = (6, 10)
MARGIN_RANGE = (1, 4)
SHAPE_COUNT_RANGE = (3, 6)
PATTERN_ATTEMPTS = 120
SHAPE_PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 300

_SWAP_TABLE = np.arange(10, dtype=int)
_SWAP_TABLE[RED] = BLUE
_SWAP_TABLE[BLUE] = RED
_SWAP_TABLE[GREEN] = YELLOW
_SWAP_TABLE[YELLOW] = GREEN
_SWAP_TABLE[ORANGE] = MAROON
_SWAP_TABLE[MAROON] = ORANGE


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(int(lo), int(hi) + 1))


def _frame_mask(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    orient: str,
    axis: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    if orient == "horizontal":
        mask[axis, c0 : c1 + 1] = True
    else:
        mask[r0 : r1 + 1, axis] = True
    return mask


def _find_frame_and_axis(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], str, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, tuple[int, int, int, int], str, int]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        pts = np.argwhere(g == int(color))
        if pts.size == 0:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not (
            np.all(g[r0, c0 : c1 + 1] == int(color))
            and np.all(g[r1, c0 : c1 + 1] == int(color))
            and np.all(g[r0 : r1 + 1, c0] == int(color))
            and np.all(g[r0 : r1 + 1, c1] == int(color))
        ):
            continue

        full_rows = [r for r in range(r0, r1 + 1) if bool(np.all(g[r, c0 : c1 + 1] == int(color)))]
        full_cols = [c for c in range(c0, c1 + 1) if bool(np.all(g[r0 : r1 + 1, c] == int(color)))]
        inner_rows = [r for r in full_rows if r not in (r0, r1)]
        inner_cols = [c for c in full_cols if c not in (c0, c1)]

        if len(inner_rows) == 1 and len(inner_cols) == 0:
            orient = "horizontal"
            axis = int(inner_rows[0])
        elif len(inner_rows) == 0 and len(inner_cols) == 1:
            orient = "vertical"
            axis = int(inner_cols[0])
        else:
            continue

        bbox = (r0, c0, r1, c1)
        if np.array_equal(g == int(color), _frame_mask(g.shape, bbox, orient, axis)):
            candidates.append((int(color), bbox, orient, axis))

    assert len(candidates) == 1, "expected exactly one framed split region"
    return candidates[0]


def _swap_colors(arr: np.ndarray) -> np.ndarray:
    return _SWAP_TABLE[np.asarray(arr, dtype=int)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the colored sample half into the blank half across the grey divider,
    swapping paired colors in the copied pattern.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, (r0, c0, r1, c1), orient, axis = _find_frame_and_axis(g)
    out = g.copy()

    if orient == "horizontal":
        top = g[r0 + 1 : axis, c0 + 1 : c1]
        bottom = g[axis + 1 : r1, c0 + 1 : c1]
        assert top.shape == bottom.shape and top.size > 0

        top_count = int(np.count_nonzero(top != BLACK))
        bottom_count = int(np.count_nonzero(bottom != BLACK))
        assert (top_count == 0) != (bottom_count == 0), "one half must be blank and the other decorated"
        assert not np.any(top == frame_color) and not np.any(bottom == frame_color)

        if top_count > 0:
            assert bool(np.all(bottom == BLACK))
            out[axis + 1 : r1, c0 + 1 : c1] = _swap_colors(np.flipud(top))
        else:
            assert bool(np.all(top == BLACK))
            out[r0 + 1 : axis, c0 + 1 : c1] = _swap_colors(np.flipud(bottom))
    else:
        left = g[r0 + 1 : r1, c0 + 1 : axis]
        right = g[r0 + 1 : r1, axis + 1 : c1]
        assert left.shape == right.shape and left.size > 0

        left_count = int(np.count_nonzero(left != BLACK))
        right_count = int(np.count_nonzero(right != BLACK))
        assert (left_count == 0) != (right_count == 0), "one half must be blank and the other decorated"
        assert not np.any(left == frame_color) and not np.any(right == frame_color)

        if left_count > 0:
            assert bool(np.all(right == BLACK))
            out[r0 + 1 : r1, axis + 1 : c1] = _swap_colors(np.fliplr(left))
        else:
            assert bool(np.all(left == BLACK))
            out[r0 + 1 : r1, c0 + 1 : axis] = _swap_colors(np.fliplr(right))

    return out


def _make_stamp(max_h: int, max_w: int) -> np.ndarray:
    kinds: list[str] = ["rect"]
    if max_w >= 2:
        kinds.append("hline")
    if max_h >= 2:
        kinds.append("vline")
    if max_h >= 2 and max_w >= 2:
        kinds.append("elbow")
    if max_h >= 3 and max_w >= 3:
        kinds.append("plus")
    if (max_h >= 2 and max_w >= 3) or (max_h >= 3 and max_w >= 2):
        kinds.append("zig")

    kind = str(np.random.choice(kinds))

    if kind == "rect":
        sh = _randint(1, min(3, max_h))
        sw = _randint(1, min(4, max_w))
        return np.ones((sh, sw), dtype=bool)

    if kind == "hline":
        return np.ones((1, _randint(2, min(5, max_w))), dtype=bool)

    if kind == "vline":
        return np.ones((_randint(2, min(5, max_h)), 1), dtype=bool)

    if kind == "elbow":
        sh = _randint(2, min(4, max_h))
        sw = _randint(2, min(4, max_w))
        stamp = np.zeros((sh, sw), dtype=bool)
        stamp[0, :] = True
        stamp[:, 0] = True
        return np.rot90(stamp, _randint(0, 3))

    if kind == "plus":
        size = 5 if min(max_h, max_w) >= 5 and bool(np.random.rand() < 0.3) else 3
        stamp = np.zeros((size, size), dtype=bool)
        mid = size // 2
        stamp[mid, :] = True
        stamp[:, mid] = True
        return stamp

    base = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    if max_h >= 2 and max_w >= 3 and max_h >= 3 and max_w >= 2:
        return np.rot90(base, _randint(0, 3))
    if max_h >= 2 and max_w >= 3:
        return base
    return np.rot90(base)


def _place_stamp(canvas: np.ndarray, stamp: np.ndarray, color: int) -> bool:
    h, w = canvas.shape
    sh, sw = stamp.shape
    for _ in range(SHAPE_PLACEMENT_ATTEMPTS):
        top = _randint(0, h - sh)
        left = _randint(0, w - sw)
        block = canvas[top : top + sh, left : left + sw]
        if bool(np.any(block[stamp] != BLACK)):
            continue
        block[stamp] = int(color)
        return True
    return False


def _make_sample_region(h: int, w: int) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        region = np.full((h, w), BLACK, dtype=int)

        required_colors = [pair[_randint(0, 1)] for pair in PAIR_GROUPS]
        if bool(np.random.rand() < 0.5):
            required_colors.append(int(np.random.choice(UNCHANGED_COLORS)))

        n_shapes = _randint(max(len(required_colors), SHAPE_COUNT_RANGE[0]), SHAPE_COUNT_RANGE[1])
        extra_pool = list(required_colors) + list(UNCHANGED_COLORS) + list(NON_FRAME_COLORS)
        colors = list(required_colors)
        while len(colors) < n_shapes:
            colors.append(int(np.random.choice(extra_pool)))

        placed_required = 0
        ok = True
        for i, color in enumerate(colors):
            placed = False
            for _ in range(SHAPE_PLACEMENT_ATTEMPTS):
                stamp = _make_stamp(h, w)
                if _place_stamp(region, stamp, int(color)):
                    placed = True
                    if i < len(required_colors):
                        placed_required += 1
                    break
            if i < len(required_colors) and not placed:
                ok = False
                break

        if not ok or placed_required != len(required_colors):
            continue

        used_colors = {int(v) for v in np.unique(region) if int(v) != BLACK}
        if len(used_colors) < len(required_colors):
            continue
        if int(np.count_nonzero(region != BLACK)) < 6:
            continue
        if int(np.count_nonzero(region != BLACK)) >= region.size:
            continue
        return region

    raise RuntimeError("failed to build a non-degenerate sample region")


def _build_input(orient: str, sample_first: bool, h: int, w: int) -> np.ndarray:
    if orient == "vertical":
        frame_h = h + 2
        frame_w = 2 * w + 3
    else:
        frame_h = 2 * h + 3
        frame_w = w + 2

    top = _randint(*MARGIN_RANGE)
    bottom = _randint(*MARGIN_RANGE)
    left = _randint(*MARGIN_RANGE)
    right = _randint(*MARGIN_RANGE)

    grid = np.full((top + frame_h + bottom, left + frame_w + right), BLACK, dtype=int)
    r0, c0 = top, left
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    sample = _make_sample_region(h, w)

    if orient == "vertical":
        axis = c0 + 1 + w
        grid[r0 : r1 + 1, axis] = GREY
        left_slice = (slice(r0 + 1, r1), slice(c0 + 1, axis))
        right_slice = (slice(r0 + 1, r1), slice(axis + 1, c1))
        if sample_first:
            grid[left_slice] = sample
        else:
            grid[right_slice] = sample
    else:
        axis = r0 + 1 + h
        grid[axis, c0 : c1 + 1] = GREY
        top_slice = (slice(r0 + 1, axis), slice(c0 + 1, c1))
        bottom_slice = (slice(axis + 1, r1), slice(c0 + 1, c1))
        if sample_first:
            grid[top_slice] = sample
        else:
            grid[bottom_slice] = sample

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        orient = "vertical" if bool(np.random.randint(0, 2)) else "horizontal"
        sample_first = bool(np.random.randint(0, 2))
        half_h = _randint(*HALF_SIZE_RANGE)
        half_w = _randint(*HALF_SIZE_RANGE)

        grid = _build_input(orient, sample_first, half_h, half_w)

        frame_color, (r0, c0, r1, c1), found_orient, axis = _find_frame_and_axis(grid)
        assert frame_color == GREY and found_orient == orient

        if orient == "vertical":
            left_half = grid[r0 + 1 : r1, c0 + 1 : axis]
            right_half = grid[r0 + 1 : r1, axis + 1 : c1]
            assert left_half.shape == (half_h, half_w)
            assert right_half.shape == (half_h, half_w)
            if sample_first:
                assert int(np.count_nonzero(left_half != BLACK)) > 0
                assert bool(np.all(right_half == BLACK))
            else:
                assert bool(np.all(left_half == BLACK))
                assert int(np.count_nonzero(right_half != BLACK)) > 0
        else:
            top_half = grid[r0 + 1 : axis, c0 + 1 : c1]
            bottom_half = grid[axis + 1 : r1, c0 + 1 : c1]
            assert top_half.shape == (half_h, half_w)
            assert bottom_half.shape == (half_h, half_w)
            if sample_first:
                assert int(np.count_nonzero(top_half != BLACK)) > 0
                assert bool(np.all(bottom_half == BLACK))
            else:
                assert bool(np.all(top_half == BLACK))
                assert int(np.count_nonzero(bottom_half != BLACK)) > 0

        return grid

    raise RuntimeError("failed to generate valid input")
