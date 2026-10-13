from __future__ import annotations

import random

import numpy as np


Frame = tuple[int, int, int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_frames(grid: np.ndarray) -> tuple[int, list[Frame]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    bg = _get_bg(g)
    assert np.all(g[1:, 1:] == bg), "only the top row and left column may contain markers"

    axis_colors = set(map(int, g[0, :])) | set(map(int, g[:, 0]))
    axis_colors.discard(bg)
    assert axis_colors, "expected at least one projected square"

    all_top_cols: list[int] = []
    all_top_rows: list[int] = []
    frames: list[Frame] = []
    for color in sorted(axis_colors):
        top_cols = np.flatnonzero(g[0, :] == color).tolist()
        left_rows = np.flatnonzero(g[:, 0] == color).tolist()
        assert len(top_cols) == 2, "each color must appear twice on the top row"
        assert len(left_rows) == 1, "each color must appear once on the left column"
        assert int(np.count_nonzero(g == color)) == 3, "markers must be axis-only and non-overlapping"

        c1, c2 = map(int, sorted(top_cols))
        r1 = int(left_rows[0])
        assert 0 < c1 < c2 < w
        assert 0 < r1 < h

        size = c2 - c1 + 1
        r2 = r1 + size - 1
        assert r2 < h, "decoded square must fit in the grid"

        frames.append((size, r1, c1, r2, c2, color))
        all_top_cols.extend([c1, c2])
        all_top_rows.append(r1)

    sizes = [frame[0] for frame in frames]
    assert len(set(sizes)) == len(sizes), "square sizes must be unique to make the layer order unique"
    assert len(set(all_top_cols)) == len(all_top_cols), "top-row markers may not share columns"
    assert len(set(all_top_rows)) == len(all_top_rows), "left-column markers may not share rows"
    return bg, frames


def _draw_frames(shape: tuple[int, int], bg: int, frames: list[Frame], *, descending: bool) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    ordered = sorted(frames, key=lambda frame: frame[0], reverse=descending)
    for _size, r1, c1, r2, c2, color in ordered:
        out[r1, c1 : c2 + 1] = color
        out[r2, c1 : c2 + 1] = color
        out[r1 : r2 + 1, c1] = color
        out[r1 : r2 + 1, c2] = color
    out[0, :] = bg
    out[:, 0] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restore nested hollow square frames from axis-only projection remnants.

    Input structure:
    - The background fills the whole grid except for markers on the top row and
      the left column.
    - Each foreground color appears exactly twice on the top row and exactly
      once on the left column.
    - Those three markers encode one missing single-cell-thick square frame of
      that color.

    Rule:
    - For each color, read its two top-row columns as `c1..c2`, derive the side
      length `s = c2 - c1 + 1`, and use its left-column row `r1` as the square's
      top edge.
    - Reconstruct the hollow square frame on rows `r1..r1+s-1` and columns
      `c1..c2`.
    - Draw all frames from larger area to smaller area so that smaller frames
      overwrite larger ones on overlapping border cells, then clear the original
      axis markers back to background.

    Key insight:
    - The input only shows two horizontal endpoints and one vertical start for
      each color; the missing lower edge is forced by the fact that the hidden
      object must be a square rather than an arbitrary rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames = _extract_frames(g)
    return _draw_frames(g.shape, bg, frames, descending=True)


GEN_BG = 0
GEN_COLOR_POOL = list(range(1, 10))
GEN_HEIGHT_RANGE = (10, 28)
GEN_WIDTH_RANGE = (10, 28)
GEN_FRAME_COUNT_RANGE = (2, 5)
GEN_MIN_SIZE = 3
GEN_MAX_ATTEMPTS = 500


def _sample_sizes(max_size: int, count: int) -> list[int]:
    sizes = sorted(random.sample(range(GEN_MIN_SIZE, max_size + 1), count), reverse=True)
    assert len(sizes) == count
    return sizes


def _sample_overlap_flags(gaps: list[int]) -> list[bool]:
    if len(gaps) == 1:
        return [True]

    mode = random.choice(("all_overlap", "mixed", "sparse_overlap"))
    flags: list[bool] = []
    for gap in gaps:
        if gap == 1:
            flags.append(True)
        elif mode == "all_overlap":
            flags.append(True)
        elif mode == "mixed":
            flags.append(random.random() < 0.55)
        else:
            flags.append(random.random() < 0.30)

    if not any(flags):
        candidates = [i for i, gap in enumerate(gaps) if gap >= 1]
        flags[random.choice(candidates)] = True

    if mode != "all_overlap" and len(flags) >= 2 and all(flags) and any(gap >= 2 for gap in gaps):
        candidates = [i for i, gap in enumerate(gaps) if gap >= 2]
        flags[random.choice(candidates)] = False

    return flags


def _sample_nested_frames(shape: tuple[int, int]) -> list[Frame] | None:
    h, w = shape
    max_count = min(
        GEN_FRAME_COUNT_RANGE[1],
        h - 1,
        (w - 1) // 2,
        min(h - 1, w - 1) - GEN_MIN_SIZE + 1,
    )
    if max_count < GEN_FRAME_COUNT_RANGE[0]:
        return None

    count = random.randint(GEN_FRAME_COUNT_RANGE[0], max_count)
    max_size = min(h - 1, w - 1)
    if max_size < GEN_MIN_SIZE + count - 1:
        return None

    sizes = _sample_sizes(max_size, count)
    gaps = [sizes[i] - sizes[i + 1] for i in range(count - 1)]
    overlap_flags = _sample_overlap_flags(gaps)

    outer_size = sizes[0]
    top0 = random.randint(1, h - outer_size)
    left0 = random.randint(1, w - outer_size)
    bottom0 = top0 + outer_size - 1
    right0 = left0 + outer_size - 1

    frames: list[Frame] = [(outer_size, top0, left0, bottom0, right0, -1)]
    for idx in range(1, count):
        prev_size, prev_top, prev_left, prev_bottom, prev_right, _prev_color = frames[-1]
        size = sizes[idx]
        share_bottom = overlap_flags[idx - 1]

        min_bottom = prev_top + size
        max_bottom = prev_bottom if share_bottom else prev_bottom - 1
        if min_bottom > max_bottom:
            return None

        min_right = prev_left + size
        max_right = prev_right - 1
        if min_right > max_right:
            return None

        bottom = random.randint(min_bottom, max_bottom)
        right = random.randint(min_right, max_right)
        top = bottom - size + 1
        left = right - size + 1
        frames.append((size, top, left, bottom, right, -1))

    top_rows = [frame[1] for frame in frames]
    endpoint_cols = []
    for _size, _top, left, _bottom, right, _color in frames:
        endpoint_cols.extend([left, right])

    if len(set(top_rows)) != len(top_rows):
        return None
    if len(set(endpoint_cols)) != len(endpoint_cols):
        return None

    for outer, inner in zip(frames, frames[1:]):
        assert outer[1] < inner[1]
        assert outer[2] < inner[2]
        assert outer[3] >= inner[3]
        assert outer[4] > inner[4]

    colors = random.sample(GEN_COLOR_POOL, count)
    return [
        (size, top, left, bottom, right, color)
        for (size, top, left, bottom, right, _), color in zip(frames, colors)
    ]


def _encode_input(shape: tuple[int, int], frames: list[Frame]) -> np.ndarray:
    h, w = shape
    inp = np.full((h, w), GEN_BG, dtype=int)
    for _size, top, left, _bottom, right, color in frames:
        inp[0, left] = color
        inp[0, right] = color
        inp[top, 0] = color
    return inp


def _check_semantics(inp: np.ndarray, frames: list[Frame], expected: np.ndarray) -> bool:
    try:
        bg, decoded = _extract_frames(inp)
    except AssertionError:
        return False

    intended = sorted(frames)
    decoded_sorted = sorted(decoded)
    if intended != decoded_sorted:
        return False

    desc = _draw_frames(inp.shape, bg, decoded, descending=True)
    asc = _draw_frames(inp.shape, bg, decoded, descending=False)
    if not np.array_equal(desc, expected):
        return False
    if np.array_equal(desc, asc):
        return False
    if not np.all(expected[0, :] == bg) or not np.all(expected[:, 0] == bg):
        return False

    bottoms = [frame[3] for frame in decoded]
    if len(set(bottoms)) == len(bottoms):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    frames = _sample_nested_frames((h, w))
    if frames is None:
        return None

    inp = _encode_input((h, w), frames)
    expected = _draw_frames((h, w), GEN_BG, frames, descending=True)

    if not _check_semantics(inp, frames, expected):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid axis-projection square-frame puzzle")
