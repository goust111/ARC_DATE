from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
OVERLAP_RED = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _interval_overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _find_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 2, "expected exactly two foreground colors"

    rects: list[dict[str, int]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert int(coords.shape[0]) == h * w, "each foreground object must be a solid rectangle"
        assert np.all(patch == color), "rectangle bbox must contain only its own color"
        rects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
            }
        )

    a, b = rects
    rows_disjoint = int(a["r1"]) < int(b["r0"]) or int(b["r1"]) < int(a["r0"])
    cols_disjoint = int(a["c1"]) < int(b["c0"]) or int(b["c1"]) < int(a["c0"])
    assert rows_disjoint and cols_disjoint, "rectangle row and column intervals must both be disjoint"
    return rects


def _direction_toward(rect: dict[str, int], other: dict[str, int]) -> tuple[int, int]:
    dr = 1 if int(other["r0"]) > int(rect["r1"]) else -1
    dc = 1 if int(other["c0"]) > int(rect["c1"]) else -1
    return int(dr), int(dc)


def _trajectory_traces(
    shape: tuple[int, int], rect: dict[str, int], dr: int, dc: int
) -> tuple[list[np.ndarray], int]:
    height, width = map(int, shape)
    r = int(rect["r0"])
    c = int(rect["c0"])
    h = int(rect["h"])
    w = int(rect["w"])

    traces: list[np.ndarray] = []
    clipped = 0
    while r < height and c < width and r + h > 0 and c + w > 0:
        rr0 = max(0, r)
        rr1 = min(height, r + h)
        cc0 = max(0, c)
        cc1 = min(width, c + w)
        if rr0 < rr1 and cc0 < cc1:
            stamp = np.zeros((height, width), dtype=bool)
            stamp[rr0:rr1, cc0:cc1] = True
            traces.append(stamp)
            if (rr1 - rr0) * (cc1 - cc0) < h * w:
                clipped += 1
        r += int(dr) * h
        c += int(dc) * w
    return traces, int(clipped)


def _union_mask(traces: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for stamp in traces:
        out |= stamp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the two solid rectangles repeatedly toward each other and mark their trajectory overlap in red.

    Input structure:
    - The background is black and the only foreground objects are two solid axis-aligned rectangles.
    - The two rectangles have different colors, and their row intervals and column intervals are both disjoint.
    - Each rectangle therefore sees the other in exactly one diagonal quadrant.

    Rule:
    - For each rectangle, keep the original copy and keep stamping the same filled rectangle only in the diagonal direction toward the other rectangle.
    - The row step equals that rectangle's own height and the column step equals its own width; any out-of-bounds part is clipped and a trajectory stops once a full stamp is outside the grid.
    - After both full trajectories are laid down, color cells covered by exactly one trajectory with that trajectory's color, and color cells covered by both trajectories red.

    Key insight:
    - Each rectangle is both the sprite and the stride, but unlike a two-way spray, each path uses exactly one diagonal aimed at the other rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    rects = _find_rectangles(g)
    a, b = rects
    da = _direction_toward(a, b)
    db = _direction_toward(b, a)

    traces_a, _ = _trajectory_traces(g.shape, a, *da)
    traces_b, _ = _trajectory_traces(g.shape, b, *db)
    mask_a = _union_mask(traces_a, g.shape)
    mask_b = _union_mask(traces_b, g.shape)

    out = np.full(g.shape, int(bg), dtype=int)
    out[mask_a & ~mask_b] = int(a["color"])
    out[mask_b & ~mask_a] = int(b["color"])
    out[mask_a & mask_b] = OVERLAP_RED
    return out


def _choose_sizes() -> tuple[int, int, int, int]:
    dims = [(h, w) for h in range(2, 6) for w in range(2, 6) if 4 <= h * w <= 16]
    h1, w1 = random.choice(dims)
    h2, w2 = random.choice(dims)
    if (h1, w1) == (h2, w2):
        return _choose_sizes()
    return int(h1), int(w1), int(h2), int(w2)


def _collision_candidates(len_a: int, len_b: int) -> list[int]:
    vals = []
    for delta in range(-(len_b - 1), len_a):
        overlap = _interval_overlap_len(0, len_a - 1, delta, delta + len_b - 1)
        if 1 <= overlap <= max(1, min(len_a, len_b) - 1):
            vals.append(int(delta))
    return vals or [0]


def _make_canonical_instance(want_clip: bool) -> tuple[np.ndarray, dict[int, int]] | None:
    h1, w1, h2, w2 = _choose_sizes()
    n, m = random.choice(((1, 1), (1, 2), (2, 1), (2, 2)))

    row_delta = random.choice(_collision_candidates(h1, h2))
    col_delta = random.choice(_collision_candidates(w1, w2))
    row_overlap = _interval_overlap_len(0, h1 - 1, row_delta, row_delta + h2 - 1)
    col_overlap = _interval_overlap_len(0, w1 - 1, col_delta, col_delta + w2 - 1)
    red_area = row_overlap * col_overlap
    if red_area < 2:
        return None

    row_gap = n * h1 + m * h2 - h1 + row_delta
    col_gap = n * w1 + m * w2 - w1 + col_delta
    if row_gap < 0 or col_gap < 0 or row_gap + col_gap == 0:
        return None

    base_h = h1 + row_gap + h2
    base_w = w1 + col_gap + w2
    if base_h > 30 or base_w > 30:
        return None

    color1, color2 = random.sample(INPUT_COLORS, 2)
    step_by_color = {int(color1): int(n), int(color2): int(m)}

    layouts: list[tuple[int, int, int, int]] = []
    if want_clip:
        for _ in range(180):
            top = random.randint(0, 30 - base_h)
            left = random.randint(0, 30 - base_w)
            max_bottom = 30 - base_h - top
            max_right = 30 - base_w - left
            bottom = random.randint(0, max_bottom)
            right = random.randint(0, max_right)
            layouts.append((top, left, base_h + top + bottom, base_w + left + right))
    else:
        min_qr = (base_h + h1 - 1) // h1
        min_qc = (base_w + w1 - 1) // w1
        top_opts = [
            top
            for top in range(0, 31 - base_h)
            if (top + h1 + row_gap) % h2 == 0
        ]
        left_opts = [
            left
            for left in range(0, 31 - base_w)
            if (left + w1 + col_gap) % w2 == 0
        ]
        random.shuffle(top_opts)
        random.shuffle(left_opts)
        for top in top_opts[:8]:
            q_rows = [
                q
                for q in range(min_qr, (30 - top) // h1 + 1)
                if top + q * h1 >= top + base_h
            ]
            if not q_rows:
                continue
            for left in left_opts[:8]:
                q_cols = [
                    q
                    for q in range(min_qc, (30 - left) // w1 + 1)
                    if left + q * w1 >= left + base_w
                ]
                if not q_cols:
                    continue
                random.shuffle(q_rows)
                random.shuffle(q_cols)
                for q_row in q_rows[:4]:
                    for q_col in q_cols[:4]:
                        layouts.append((top, left, top + q_row * h1, left + q_col * w1))

    random.shuffle(layouts)
    for top, left, height, width in layouts:
        grid = np.full((height, width), BACKGROUND, dtype=int)
        r1 = int(top)
        c1 = int(left)
        r2 = int(top + h1 + row_gap)
        c2 = int(left + w1 + col_gap)
        grid[r1 : r1 + h1, c1 : c1 + w1] = int(color1)
        grid[r2 : r2 + h2, c2 : c2 + w2] = int(color2)
        if _validate_instance(grid, step_by_color, require_clip=want_clip):
            return grid, step_by_color
    return None


def _flip_grid(grid: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = grid
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _validate_instance(grid: np.ndarray, step_by_color: dict[int, int], require_clip: bool) -> bool:
    try:
        rects = _find_rectangles(grid)
    except AssertionError:
        return False

    rect_by_color = {int(rect["color"]): rect for rect in rects}
    if set(rect_by_color) != set(step_by_color):
        return False

    masks: dict[int, np.ndarray] = {}
    traces_by_color: dict[int, list[np.ndarray]] = {}
    clipped_total = 0
    for color, rect in rect_by_color.items():
        other_color = next(c for c in rect_by_color if c != color)
        dr, dc = _direction_toward(rect, rect_by_color[other_color])
        traces, clipped = _trajectory_traces(grid.shape, rect, dr, dc)
        traces_by_color[int(color)] = traces
        masks[int(color)] = _union_mask(traces, grid.shape)
        clipped_total += int(clipped)

    out = main(grid)
    red_count = int(np.count_nonzero(out == OVERLAP_RED))
    if red_count < 2:
        return False
    if int(np.count_nonzero(out != BACKGROUND)) > int(grid.size * 0.48):
        return False

    for color, rect in rect_by_color.items():
        traces = traces_by_color[int(color)]
        needed_step = int(step_by_color[int(color)])
        if len(traces) < needed_step + 2:
            return False
        other_color = next(c for c in rect_by_color if c != color)
        other_mask = masks[int(other_color)]
        future_exclusive = any(np.any(stamp & ~other_mask) for stamp in traces[needed_step + 1 :])
        if not future_exclusive:
            return False
        if int(np.count_nonzero(out == int(color))) <= int(rect["area"]):
            return False

    if require_clip and clipped_total == 0:
        return False
    if not require_clip and clipped_total != 0:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        want_clip = random.random() < 0.6
        built = _make_canonical_instance(want_clip)
        if built is None:
            continue
        grid, step_by_color = built
        flip_ud = bool(random.getrandbits(1))
        flip_lr = bool(random.getrandbits(1))
        grid = _flip_grid(grid, flip_ud=flip_ud, flip_lr=flip_lr)
        if _validate_instance(grid, step_by_color, require_clip=want_clip):
            return grid
    raise RuntimeError("failed to generate a valid ballistic-collision input")
