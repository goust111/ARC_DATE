from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _seed_bounds(grid: np.ndarray) -> tuple[int, int]:
    row_has_fg = np.any(grid != BG, axis=1)
    assert bool(row_has_fg[-1]), "seed motif must reach the bottom row"
    last = int(np.flatnonzero(row_has_fg)[-1])
    top = last
    while top > 0 and bool(row_has_fg[top - 1]):
        top -= 1
    return top, last


def _frame_info(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    frame_mask = grid == GREEN
    frame_bbox = _bbox(frame_mask)
    assert frame_bbox is not None, "expected a green hollow rectangle"
    r0, c0, r1, c1 = frame_bbox
    assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5

    expected = np.zeros_like(frame_mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(frame_mask, expected), "green pixels must form one hollow rectangle"

    interior = np.zeros_like(frame_mask, dtype=bool)
    interior[r0 + 1 : r1, c0 + 1 : c1] = True
    return frame_bbox, frame_mask, interior


def _window_info(grid: np.ndarray, interior: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    marker_mask = grid == YELLOW
    window_bbox = _bbox(marker_mask)
    assert window_bbox is not None, "expected yellow marker pixels"
    wr0, wc0, wr1, wc1 = window_bbox
    assert np.all(interior[marker_mask]), "markers must lie inside the frame interior"

    ir0, ic0, ir1, ic1 = _bbox(interior)
    assert ir0 is not None
    assert wr0 > ir0 and wc0 > ic0 and wr1 < ir1 and wc1 < ic1, "window must sit strictly inside the interior"

    window_mask = np.zeros_like(marker_mask, dtype=bool)
    window_mask[wr0 : wr1 + 1, wc0 : wc1 + 1] = True
    assert np.any(window_mask & ~marker_mask), "window must contain non-marker cells"
    return window_bbox, window_mask, marker_mask


def _tiled_seed(seed: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    period = int(seed.shape[0])
    height, _width = out_shape
    out = np.empty(out_shape, dtype=int)
    for r in range(height):
        out[r] = seed[r % period]
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    seed_top, seed_bottom = _seed_bounds(g)
    seed = g[seed_top : seed_bottom + 1].copy()

    frame_bbox, frame_mask, interior = _frame_info(g)
    _window_bbox, window_mask, marker_mask = _window_info(g, interior)
    _ = frame_bbox

    tiled = _tiled_seed(seed, g.shape)
    out = tiled.copy()
    out[interior & ~window_mask] = BG
    out[frame_mask] = GREEN
    out[marker_mask] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the bottom seed upward, but blank a protected hollow rectangle except for a marker-defined window.

    Input structure:
    - A black background contains a contiguous bottom row block made from blue and red seed pixels, with background
      holes allowed inside those rows.
    - Higher in the grid, green pixels form one one-cell-thick hollow rectangle whose interior is otherwise black.
    - A few yellow marker pixels lie inside that rectangle; their bounding box defines a smaller rectangular window.

    Rule:
    - Read the bottom contiguous non-background row block as the seed motif with period P.
    - Fill the whole grid by repeating seed row `r mod P` into every row `r`.
    - Then enforce the protection priority: keep the green frame green, reset the frame interior outside the marker
      window to black, keep the window's tiled result visible, and finally restore the yellow marker pixels themselves.

    Key insight:
    - The output is a conflict-resolution overlay, not a simple copy: global periodic tiling happens everywhere first,
      then the hollow rectangle carves out a protected blank region, except that the markers reopen one transparent
      rectangular window inside it.
    """
    return _solve(grid)


def _seed_has_shorter_period(seed: np.ndarray) -> bool:
    period = int(seed.shape[0])
    for k in range(1, period):
        if period % k == 0 and np.array_equal(seed, seed[np.arange(period) % k]):
            return True
    return False


def _make_seed(height: int, width: int) -> np.ndarray:
    seed = np.zeros((height, width), dtype=int)
    for r in range(height):
        row_kind = random.choice(["mixed", "band", "sparse"])
        if row_kind == "band":
            span = random.randint(max(3, width // 3), width)
            c0 = random.randint(0, width - span)
            seed[r, c0 : c0 + span] = random.choice([BLUE, RED])
            for c in range(c0, c0 + span):
                if random.random() < 0.25:
                    seed[r, c] = random.choice([BLUE, RED])
                if random.random() < 0.18:
                    seed[r, c] = BG
        elif row_kind == "sparse":
            cols = random.sample(range(width), random.randint(max(3, width // 3), max(4, width - 1)))
            for c in cols:
                seed[r, c] = random.choice([BLUE, RED])
        else:
            for c in range(width):
                roll = random.random()
                if roll < 0.34:
                    seed[r, c] = BG
                elif roll < 0.67:
                    seed[r, c] = BLUE
                else:
                    seed[r, c] = RED

    if not np.any(seed == BLUE) or not np.any(seed == RED):
        rr = random.randrange(height)
        cc = random.randrange(width)
        seed[rr, cc] = BLUE
        seed[(rr + 1) % height, (cc + 1) % width] = RED
    for r in range(height):
        if not np.any(seed[r] != BG):
            cols = random.sample(range(width), 2)
            seed[r, cols[0]] = BLUE
            seed[r, cols[1]] = RED
    if not np.any(seed == BG):
        seed[random.randrange(height), random.randrange(width)] = BG
    if len({tuple(row.tolist()) for row in seed}) < height:
        row = random.randrange(height)
        col = random.randrange(width)
        seed[row, col] = BLUE if seed[row, col] != BLUE else RED
    return seed


def _marker_points_for_window(wr0: int, wc0: int, wr1: int, wc1: int, count: int) -> list[tuple[int, int]]:
    anchors = {(wr0, wc0), (wr1, wc1)}
    if count >= 3:
        anchors.add((wr0, wc1))
    if count >= 4:
        anchors.add((wr1, wc0))

    candidates = [
        (r, c)
        for r in range(wr0, wr1 + 1)
        for c in range(wc0, wc1 + 1)
        if (r, c) not in anchors
    ]
    points = list(anchors)
    random.shuffle(points)
    while len(points) < count and candidates:
        idx = random.randrange(len(candidates))
        points.append(candidates.pop(idx))
    return points[:count]


def _sample_dimensions(period: int) -> tuple[int, int]:
    height_choices = [h for h in range(12, 25) if h % period == 0]
    height = random.choice(height_choices)
    width = random.randint(10, 18)
    return height, width


def _sample_window_size(max_h: int, max_w: int) -> tuple[int, int]:
    if random.random() < 0.35:
        if max_w >= 2 and (max_h == 1 or random.random() < 0.5):
            return 1, random.randint(2, max_w)
        if max_h >= 2:
            return random.randint(2, max_h), 1
    return random.randint(1, max_h), random.randint(1, max_w)


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        seed_top, seed_bottom = _seed_bounds(inp)
        seed = inp[seed_top : seed_bottom + 1].copy()
        frame_bbox, frame_mask, interior = _frame_info(inp)
        _window_bbox, window_mask, marker_mask = _window_info(inp, interior)
    except AssertionError:
        return False

    if _seed_has_shorter_period(seed):
        return False

    tiled = _tiled_seed(seed, inp.shape)
    if not np.array_equal(out, _solve(inp)):
        return False

    ir = interior & ~window_mask
    if not np.any(ir):
        return False
    if not np.all(out[ir] == BG):
        return False
    if not np.all(out[frame_mask] == GREEN):
        return False
    if not np.all(out[marker_mask] == YELLOW):
        return False
    if not np.array_equal(out[window_mask & ~marker_mask], tiled[window_mask & ~marker_mask]):
        return False

    if not np.any((window_mask & ~marker_mask) & (tiled != BG)):
        return False
    if not np.any(ir & (tiled != BG)):
        return False
    if not np.any((window_mask & ~marker_mask) & (tiled == BG)):
        return False

    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    if not np.array_equal(out[outside], tiled[outside]):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    period = random.randint(2, 4)
    height, width = _sample_dimensions(period)

    seed = _make_seed(period, width)
    if _seed_has_shorter_period(seed):
        return None

    max_frame_h = height - period - 2
    if max_frame_h < 5 or width < 7:
        return None
    frame_h = random.randint(5, min(9, max_frame_h))
    frame_w = random.randint(5, min(width - 2, 12))
    frame_r0_max = height - period - frame_h - 1
    if frame_r0_max < 1:
        return None
    frame_r0 = random.randint(1, frame_r0_max)
    center_pref = max(1, (width - frame_w) // 2)
    left_lo = max(1, center_pref - 2)
    left_hi = min(width - frame_w - 1, center_pref + 2)
    if left_lo > left_hi:
        left_lo = 1
        left_hi = width - frame_w - 1
    if left_lo > left_hi:
        return None
    frame_c0 = random.randint(left_lo, left_hi)
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1

    interior_h = frame_h - 2
    interior_w = frame_w - 2
    max_window_h = interior_h - 2
    max_window_w = interior_w - 2
    if max_window_h < 1 or max_window_w < 1:
        return None
    window_h, window_w = _sample_window_size(max_window_h, max_window_w)
    window_area = window_h * window_w
    if window_area < 3:
        return None
    wr0 = random.randint(frame_r0 + 2, frame_r1 - 1 - window_h)
    wc0 = random.randint(frame_c0 + 2, frame_c1 - 1 - window_w)
    wr1 = wr0 + window_h - 1
    wc1 = wc0 + window_w - 1

    marker_count = random.randint(2, min(4, window_area - 1))
    if window_area <= marker_count:
        return None
    markers = _marker_points_for_window(wr0, wc0, wr1, wc1, marker_count)
    if len(set(markers)) != marker_count:
        return None

    inp = np.zeros((height, width), dtype=int)
    inp[height - period : height] = seed
    inp[frame_r0, frame_c0 : frame_c1 + 1] = GREEN
    inp[frame_r1, frame_c0 : frame_c1 + 1] = GREEN
    inp[frame_r0 : frame_r1 + 1, frame_c0] = GREEN
    inp[frame_r0 : frame_r1 + 1, frame_c1] = GREEN
    for r, c in markers:
        inp[r, c] = YELLOW

    out = main(inp)
    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
