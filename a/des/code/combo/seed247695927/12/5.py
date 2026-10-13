from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    h, w = grid.shape
    found: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = np.zeros((h, w), dtype=bool)
        expected[r0 : r1 + 1, c0] = True
        expected[r0 : r1 + 1, c1] = True
        expected[r0, c0 : c1 + 1] = True
        expected[r1, c0 : c1 + 1] = True
        if np.array_equal(grid == color, expected):
            found.append((color, (r0, c0, r1, c1)))
    assert len(found) == 1, "expected a unique rectangular frame"
    return found[0]


def _full_rows(mask: np.ndarray) -> list[int]:
    return [int(r) for r in range(mask.shape[0]) if bool(mask[r, :].all())]


def _full_cols(mask: np.ndarray) -> list[int]:
    return [int(c) for c in range(mask.shape[1]) if bool(mask[:, c].all())]


def _is_arithmetic(seq: list[int]) -> bool:
    if len(seq) < 2:
        return False
    step = seq[1] - seq[0]
    if step <= 0:
        return False
    return all(seq[i + 1] - seq[i] == step for i in range(len(seq) - 1))


def _detect_pattern(
    interior: np.ndarray, colors: list[int]
) -> tuple[str, int, int, list[int], int, int]:
    for marker_color in colors:
        stripe_color = next(color for color in colors if color != marker_color)
        marker_mask = interior == marker_color
        stripe_mask = interior == stripe_color
        rows = _full_rows(marker_mask)
        cols = _full_cols(marker_mask)

        if len(rows) == 2:
            top, bottom = rows
            valid = [r for r in _full_rows(stripe_mask) if top < r < bottom]
            if len(valid) >= 2:
                valid = sorted(valid)
                if _is_arithmetic(valid):
                    return "horizontal", marker_color, stripe_color, valid, top, bottom

        if len(cols) == 2:
            left, right = cols
            valid = [c for c in _full_cols(stripe_mask) if left < c < right]
            if len(valid) >= 2:
                valid = sorted(valid)
                if _is_arithmetic(valid):
                    return "vertical", marker_color, stripe_color, valid, left, right

    raise AssertionError("failed to detect marker bars and periodic stripes")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the periodic stripe pattern inside a frame across the canvas.

    Input structure:
    - A background canvas contains a one-pixel-thick monochrome rectangular frame.
    - Inside the frame are two parallel full marker bars of another color.
    - A third color fills periodic rows or columns between the marker bars.

    Rule:
    - Infer the periodic row/column mask from the third-color stripes.
    - Paint the frame color along those full rows/columns across the whole canvas.
    - Extend the marker bars across the whole canvas.
    - Repaint the source stripe color between the marker bars on the periodic rows/columns.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, (r0, c0, r1, c1) = _find_frame(g, bg)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(v) for v in np.unique(interior) if int(v) != bg]
    assert len(colors) == 2, "expected one marker color and one stripe color inside the frame"

    orientation, marker_color, stripe_color, stripe_positions, marker_a, marker_b = _detect_pattern(interior, colors)
    step = stripe_positions[1] - stripe_positions[0]

    out = np.full_like(g, bg)
    h, w = g.shape

    if orientation == "horizontal":
        period_rows = [r for r in range(h) if (r - (r0 + 1 + stripe_positions[0])) % step == 0]
        for r in period_rows:
            out[r, :] = frame_color
        out[r0 + 1 + marker_a, :] = marker_color
        out[r0 + 1 + marker_b, :] = marker_color
        for r in period_rows:
            if r0 + 1 + marker_a < r < r0 + 1 + marker_b:
                out[r, :] = stripe_color
        out[r0 + 1 + marker_a, :] = marker_color
        out[r0 + 1 + marker_b, :] = marker_color
    else:
        period_cols = [c for c in range(w) if (c - (c0 + 1 + stripe_positions[0])) % step == 0]
        for c in period_cols:
            out[:, c] = frame_color
        out[:, c0 + 1 + marker_a] = marker_color
        out[:, c0 + 1 + marker_b] = marker_color
        for c in period_cols:
            if c0 + 1 + marker_a < c < c0 + 1 + marker_b:
                out[:, c] = stripe_color
        out[:, c0 + 1 + marker_a] = marker_color
        out[:, c0 + 1 + marker_b] = marker_color

    return out


def _sample_axis_params(axis_len: int) -> tuple[int, int, int, int, int]:
    period = random.randint(2, 4)
    stripe_count = random.randint(2, 4)
    gap_before = random.randint(1, period)
    gap_after = random.randint(1, period)
    prefix = random.randint(0, 2)
    used = 1 + gap_before + (stripe_count - 1) * period + gap_after + 1
    remaining = axis_len - used - prefix
    if remaining < 0:
        raise ValueError("axis too short")
    suffix = random.randint(0, min(2, remaining))
    start = prefix
    first_stripe = start + gap_before
    last_stripe = first_stripe + (stripe_count - 1) * period
    end = last_stripe + gap_after
    if end >= axis_len:
        raise ValueError("markers out of bounds")
    return start, end, first_stripe, period, stripe_count


def _render_output(
    shape: tuple[int, int],
    frame_color: int,
    marker_color: int,
    stripe_color: int,
    orientation: str,
    marker_a_abs: int,
    marker_b_abs: int,
    first_abs: int,
    period: int,
) -> np.ndarray:
    h, w = shape
    out = np.full(shape, _get_bg(np.zeros(shape, dtype=int)), dtype=int)
    out.fill(0)
    if orientation == "horizontal":
        period_rows = [r for r in range(h) if (r - first_abs) % period == 0]
        for r in period_rows:
            out[r, :] = frame_color
        for r in period_rows:
            if marker_a_abs < r < marker_b_abs:
                out[r, :] = stripe_color
        out[marker_a_abs, :] = marker_color
        out[marker_b_abs, :] = marker_color
    else:
        period_cols = [c for c in range(w) if (c - first_abs) % period == 0]
        for c in period_cols:
            out[:, c] = frame_color
        for c in period_cols:
            if marker_a_abs < c < marker_b_abs:
                out[:, c] = stripe_color
        out[:, marker_a_abs] = marker_color
        out[:, marker_b_abs] = marker_color
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    bg, frame_color, marker_color, stripe_color = random.sample(range(10), 4)
    orientation = random.choice(["horizontal", "vertical"])

    h = random.randint(11, 19)
    w = random.randint(11, 19)
    grid = np.full((h, w), bg, dtype=int)

    if orientation == "horizontal":
        frame_h = random.randint(7, min(13, h - 2))
        frame_w = random.randint(5, min(12, w - 2))
    else:
        frame_h = random.randint(5, min(12, h - 2))
        frame_w = random.randint(7, min(13, w - 2))

    r0 = random.randint(1, h - frame_h - 1)
    c0 = random.randint(1, w - frame_w - 1)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color

    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1

    if orientation == "horizontal":
        axis_len = interior_h
        cross_len = interior_w
        start, end, first, period, stripe_count = _sample_axis_params(axis_len)
        marker_rows = [start, end]
        stripe_rows = [first + k * period for k in range(stripe_count)]
        for rr in marker_rows:
            grid[r0 + 1 + rr, c0 + 1 : c1] = marker_color
        for rr in stripe_rows:
            grid[r0 + 1 + rr, c0 + 1 : c1] = stripe_color
        assert cross_len >= 3
        out = np.full((h, w), bg, dtype=int)
        period_rows = [r for r in range(h) if (r - (r0 + 1 + first)) % period == 0]
        for r in period_rows:
            out[r, :] = frame_color
        for r in period_rows:
            if r0 + 1 + start < r < r0 + 1 + end:
                out[r, :] = stripe_color
        out[r0 + 1 + start, :] = marker_color
        out[r0 + 1 + end, :] = marker_color
    else:
        axis_len = interior_w
        cross_len = interior_h
        start, end, first, period, stripe_count = _sample_axis_params(axis_len)
        marker_cols = [start, end]
        stripe_cols = [first + k * period for k in range(stripe_count)]
        for cc in marker_cols:
            grid[r0 + 1 : r1, c0 + 1 + cc] = marker_color
        for cc in stripe_cols:
            grid[r0 + 1 : r1, c0 + 1 + cc] = stripe_color
        assert cross_len >= 3
        out = np.full((h, w), bg, dtype=int)
        period_cols = [c for c in range(w) if (c - (c0 + 1 + first)) % period == 0]
        for c in period_cols:
            out[:, c] = frame_color
        for c in period_cols:
            if c0 + 1 + start < c < c0 + 1 + end:
                out[:, c] = stripe_color
        out[:, c0 + 1 + start] = marker_color
        out[:, c0 + 1 + end] = marker_color

    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            inp, out = _build_case()
        except Exception:
            continue
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
