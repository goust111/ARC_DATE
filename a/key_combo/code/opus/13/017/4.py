from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _find_separator_col(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if not candidates:
        raise ValueError("expected a full-height grey separator column")
    return int(candidates[0])


def _extract_first_thickness(left: np.ndarray) -> dict[int, int]:
    h, w = left.shape
    first_thickness: dict[int, int] = {}
    r = 0
    while r < h:
        color = int(left[r, 0])
        if not np.all(left[r, :] == color):
            raise ValueError("left region must be horizontal full-width stripes")
        r2 = r + 1
        while r2 < h and int(left[r2, 0]) == color:
            if not np.all(left[r2, :] == color):
                raise ValueError("left region must be horizontal full-width stripes")
            r2 += 1
        if color not in first_thickness:
            first_thickness[color] = r2 - r
        r = r2
    return first_thickness


def _extract_bars(right: np.ndarray) -> list[dict[str, int]]:
    h, w = right.shape
    bars: list[dict[str, int]] = []
    for c in range(w):
        r = 0
        while r < h:
            color = int(right[r, c])
            if color == BLACK:
                r += 1
                continue
            r2 = r + 1
            while r2 < h and int(right[r2, c]) == color:
                r2 += 1
            bars.append(
                {
                    "color": color,
                    "col": c,
                    "top": r,
                    "bottom": r2 - 1,
                }
            )
            r = r2
    return bars


def _slide_bars(right: np.ndarray, first_thickness: dict[int, int]) -> np.ndarray:
    h, w = right.shape
    out = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    bars = sorted(_extract_bars(right), key=lambda bar: (bar["top"], bar["col"]))
    for bar in bars:
        color = bar["color"]
        if color not in first_thickness:
            raise ValueError(f"bar color {color} missing from stripe code")
        shift = int(first_thickness[color])
        top = int(bar["top"])
        bottom = int(bar["bottom"])
        col = int(bar["col"])

        # Move upward one row at a time so earlier bars block later ones.
        for _ in range(shift):
            if top == 0:
                break
            if occupied[top - 1 : bottom, col].any():
                break
            top -= 1
            bottom -= 1

        out[top : bottom + 1, col] = color
        occupied[top : bottom + 1, col] = True

    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    sep = _find_separator_col(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    first_thickness = _extract_first_thickness(left)
    return _slide_bars(right, first_thickness)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each right-side vertical bar upward by the thickness encoded by its first matching left stripe.

    Input structure:
    - A black background grid is split by one full-height grey separator column.
    - The left region is filled by full-width horizontal color stripes, possibly repeating colors in non-adjacent bands.
    - The right region contains one-cell-wide colored vertical bars on a black background.

    Rule:
    - Read the left stripes top to bottom and record, for each color, the thickness of its first stripe occurrence.
    - For every right-side bar, look up the first-occurrence thickness of the bar's color and slide that bar upward by that many rows.
    - Process bars from higher starting top row to lower; if a bar's next upward step would hit an already moved bar, stop it there.
    - Output only the transformed right region, excluding the left stripes and the grey separator.

    Key insight:
    - The compressed stripe order reveals which colors participate, but the true motion parameter is the original thickness of each color's first stripe before compression.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    return _solve(g)


def _random_partition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    if parts * lo > total or parts * hi < total:
        return None
    out: list[int] = []
    remaining = total
    for i in range(parts):
        rest = parts - i - 1
        cur_lo = max(lo, remaining - rest * hi)
        cur_hi = min(hi, remaining - rest * lo)
        if cur_lo > cur_hi:
            return None
        value = random.randint(cur_lo, cur_hi)
        out.append(value)
        remaining -= value
    return out if remaining == 0 else None


def _sample_bar_columns(width: int, count: int) -> list[int] | None:
    for _ in range(80):
        cols = sorted(random.sample(range(width), count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _sample_stripe_layout(height: int) -> tuple[list[int], list[int], dict[int, int]] | None:
    num_colors = random.randint(3, 5)
    colors = random.sample(PALETTE, num_colors)

    extra = random.choices([0, 1, 2, 3], weights=[1, 3, 3, 2])[0]
    stripe_count = min(height, num_colors + extra)
    if stripe_count < num_colors:
        return None

    for _ in range(120):
        seq = random.sample(colors, len(colors))
        while len(seq) < stripe_count:
            idx = random.randint(0, len(seq))
            left = seq[idx - 1] if idx > 0 else None
            right = seq[idx] if idx < len(seq) else None
            choices = [c for c in colors if c != left and c != right]
            if not choices:
                break
            seq.insert(idx, random.choice(choices))
        if len(seq) != stripe_count:
            continue

        thicknesses = _random_partition(height, stripe_count, 1, 4)
        if thicknesses is None:
            continue

        first_thickness: dict[int, int] = {}
        repeat_with_diff = False
        for color, thickness in zip(seq, thicknesses):
            if color in first_thickness and first_thickness[color] != thickness:
                repeat_with_diff = True
            first_thickness.setdefault(color, thickness)

        if stripe_count > num_colors and not repeat_with_diff:
            continue
        return seq, thicknesses, first_thickness

    return None


def _build_input_grid(
    height: int,
    left_w: int,
    right_w: int,
    stripe_colors: list[int],
    stripe_thicknesses: list[int],
    unique_colors: list[int],
    bar_cols: list[int],
    bar_heights: list[int],
) -> np.ndarray:
    grid = np.full((height, left_w + 1 + right_w), BLACK, dtype=int)

    r0 = 0
    for color, thickness in zip(stripe_colors, stripe_thicknesses):
        grid[r0 : r0 + thickness, :left_w] = color
        r0 += thickness

    grid[:, left_w] = GREY

    for color, col, bar_h in zip(unique_colors, bar_cols, bar_heights):
        grid[height - bar_h : height, left_w + 1 + col] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 15)
    stripe_layout = _sample_stripe_layout(height)
    if stripe_layout is None:
        return None
    stripe_colors, stripe_thicknesses, first_thickness = stripe_layout
    unique_colors = list(first_thickness.keys())
    num_bars = len(unique_colors)

    left_w = random.randint(6, 10)
    right_min = 2 * num_bars - 1
    right_max = min(12, 30 - left_w - 1)
    if right_min > right_max:
        return None
    right_w = random.randint(right_min, right_max)

    bar_cols = _sample_bar_columns(right_w, num_bars)
    if bar_cols is None:
        return None

    bar_heights = [random.randint(2, 4) for _ in range(num_bars)]
    if len(set(bar_heights)) == 1:
        idx = random.randrange(num_bars)
        bar_heights[idx] = 2 if bar_heights[idx] > 2 else 4

    grid = _build_input_grid(
        height=height,
        left_w=left_w,
        right_w=right_w,
        stripe_colors=stripe_colors,
        stripe_thicknesses=stripe_thicknesses,
        unique_colors=unique_colors,
        bar_cols=bar_cols,
        bar_heights=bar_heights,
    )

    out = main(grid)
    if out.shape != (height, right_w):
        return None

    # Reject degenerate samples that do not visibly use the code region.
    moved = False
    for color, col, bar_h in zip(unique_colors, bar_cols, bar_heights):
        shift = first_thickness[color]
        src_top = height - bar_h
        dst_rows = np.flatnonzero(out[:, col] == color)
        if dst_rows.size != bar_h:
            return None
        if int(dst_rows.min()) != src_top - shift:
            return None
        if shift > 0:
            moved = True
    if not moved:
        return None

    if len(set(unique_colors)) < 3:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
