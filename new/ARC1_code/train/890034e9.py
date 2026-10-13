from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_source_frame(g: np.ndarray) -> tuple[int, int, tuple[int, int, int, int]]:
    frames: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        coords = np.argwhere(g == color)

        r0, c0, r1, c1 = bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        interior = sub[~border]
        interior_colors = np.unique(interior)
        if (
            np.all(sub[border] == color)
            and interior_colors.size == 1
            and int(interior_colors[0]) != color
            and coords.shape[0] == int(border.sum())
        ):
            frames.append((color, int(interior_colors[0]), (r0, c0, r1, c1)))

    assert len(frames) == 1, "expected exactly one complete monochrome source frame"
    return frames[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the color of the unique complete rectangular frame onto the matching empty rectangle elsewhere.

    Input structure:
    - The grid contains one complete monochrome rectangular frame.
    - That frame's interior is a single empty color.
    - Elsewhere, exactly one same-size rectangle has the same empty-colored interior
      but a non-frame-colored perimeter.

    Rule:
    - Find the complete source frame and infer its color, size, and empty interior color.
    - Find the unique same-size non-source window whose interior is entirely the
      empty color.
    - Recolor that window's perimeter to the source frame color, preserving every
      other pixel.

    Key insight:
    - The source frame is a template for completing the camouflaged rectangle:
      its interior identifies the empty color, while its bbox gives the destination
      size and perimeter shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frame_color, empty_color, source_bbox = _find_source_frame(g)
    sr0, sc0, sr1, sc1 = source_bbox
    height = sr1 - sr0 + 1
    width = sc1 - sc0 + 1
    border = _border_mask(height, width)

    targets: list[tuple[int, int]] = []
    grid_h, grid_w = map(int, g.shape)
    for r0 in range(grid_h - height + 1):
        for c0 in range(grid_w - width + 1):
            bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
            if bbox == source_bbox:
                continue
            sub = g[r0 : r0 + height, c0 : c0 + width]
            if not np.all(sub[~border] == empty_color):
                continue
            perimeter = sub[border]
            if np.any(perimeter == frame_color):
                continue
            if not np.any(perimeter != empty_color):
                continue
            targets.append((r0, c0))

    assert len(targets) == 1, "expected exactly one same-size empty-interior target"
    target_r0, target_c0 = targets[0]

    out = g.copy()
    window = out[target_r0 : target_r0 + height, target_c0 : target_c0 + width]
    window[border] = frame_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and color IDs.
_GEN_GRID_SIZE = 21
_GEN_EMPTY = 0
_GEN_COLOR_POOL = np.arange(1, 10)
_GEN_PALETTE_SIZE = 3

# Source and target frame dimensions.
_GEN_FRAME_SIZE_RANGE = (4, 8)

# Background probability and post-stamp balance bounds.
_GEN_EMPTY_PROB_RANGE = (0.12, 0.55)
_GEN_NOISE_PROB_MIN, _GEN_NOISE_PROB_MAX = 0.12, 0.62
_GEN_BALANCE_MIN, _GEN_BALANCE_MAX = 0.08, 0.62

# Generation retry budget.
_GEN_MAX_ATTEMPTS = 300

_GenBox = tuple[int, int, int, int]
_GenNoiseColors = tuple[int, int]

_gen_border_mask = _border_mask


def _gen_palette(rng: np.random.Generator) -> tuple[int, _GenNoiseColors]:
    colors = rng.choice(_GEN_COLOR_POOL, size=_GEN_PALETTE_SIZE, replace=False)
    frame_color, noise_1, noise_2 = map(int, colors)
    return frame_color, (noise_1, noise_2)


def _gen_boxes_overlap(a: _GenBox, b: _GenBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_place_pair(
    height: int, width: int, rng: np.random.Generator
) -> tuple[_GenBox, _GenBox]:
    tops = [
        (r, c)
        for r in range(_GEN_GRID_SIZE - height + 1)
        for c in range(_GEN_GRID_SIZE - width + 1)
    ]
    sr, sc = tops[int(rng.integers(len(tops)))]
    source = (sr, sc, sr + height - 1, sc + width - 1)
    target_tops = []
    for tr, tc in tops:
        target = (tr, tc, tr + height - 1, tc + width - 1)
        if not _gen_boxes_overlap(source, target):
            target_tops.append((tr, tc))
    tr, tc = target_tops[int(rng.integers(len(target_tops)))]
    return source, (tr, tc, tr + height - 1, tc + width - 1)


def _gen_noise_probabilities(rng: np.random.Generator) -> np.ndarray:
    p0 = float(rng.uniform(*_GEN_EMPTY_PROB_RANGE))
    lo = max(_GEN_NOISE_PROB_MIN, 1.0 - p0 - _GEN_NOISE_PROB_MAX)
    hi = min(_GEN_NOISE_PROB_MAX, 1.0 - p0 - _GEN_NOISE_PROB_MIN)
    p1 = float(rng.uniform(lo, hi))
    return np.array([p0, p1, 1.0 - p0 - p1], dtype=float)


def _gen_noise_grid(noise_colors: _GenNoiseColors, rng: np.random.Generator) -> np.ndarray:
    values = np.array([_GEN_EMPTY, noise_colors[0], noise_colors[1]], dtype=int)
    return rng.choice(values, size=(_GEN_GRID_SIZE, _GEN_GRID_SIZE), p=_gen_noise_probabilities(rng))


def _gen_stamp_frame(
    grid: np.ndarray,
    box: _GenBox,
    color: int,
    border: np.ndarray,
) -> None:
    r0, c0, r1, c1 = box
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    window[:, :] = _GEN_EMPTY
    window[border] = color


def _gen_target_tops(
    grid: np.ndarray,
    height: int,
    width: int,
    source: _GenBox,
    frame_color: int,
) -> list[tuple[int, int]]:
    border = _gen_border_mask(height, width)
    tops: list[tuple[int, int]] = []
    for r0 in range(grid.shape[0] - height + 1):
        for c0 in range(grid.shape[1] - width + 1):
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if box == source:
                continue
            window = grid[r0 : r0 + height, c0 : c0 + width]
            if not np.all(window[~border] == _GEN_EMPTY):
                continue
            perimeter = window[border]
            if np.any(perimeter == frame_color):
                continue
            if np.any(perimeter != _GEN_EMPTY):
                tops.append((r0, c0))
    return tops


def _gen_break_extra_windows(
    grid: np.ndarray,
    height: int,
    width: int,
    source: _GenBox,
    target: _GenBox,
    frame_color: int,
    noise_colors: _GenNoiseColors,
    rng: np.random.Generator,
) -> bool:
    border = _gen_border_mask(height, width)
    interior_coords = np.argwhere(~border)
    protected = np.zeros(grid.shape, dtype=bool)

    sr0, sc0, sr1, sc1 = source
    protected[sr0 : sr1 + 1, sc0 : sc1 + 1] = True

    tr0, tc0, tr1, tc1 = target
    target_window_protected = protected[tr0 : tr1 + 1, tc0 : tc1 + 1]
    target_window_protected[~border] = True

    for r0 in range(grid.shape[0] - height + 1):
        for c0 in range(grid.shape[1] - width + 1):
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if box == source or box == target:
                continue
            window = grid[r0 : r0 + height, c0 : c0 + width]
            if not np.all(window[~border] == _GEN_EMPTY):
                continue
            if np.any(window[border] == frame_color):
                continue

            order = rng.permutation(len(interior_coords))
            fixed = False
            for idx in order:
                dr, dc = map(int, interior_coords[int(idx)])
                rr, cc = r0 + dr, c0 + dc
                if protected[rr, cc]:
                    continue
                grid[rr, cc] = int(noise_colors[int(rng.integers(len(noise_colors)))])
                fixed = True
                break
            if not fixed:
                return False

    target_top = (target[0], target[1])
    return _gen_target_tops(grid, height, width, source, frame_color) == [target_top]


def _gen_color_balance_ok(
    grid: np.ndarray,
    frame_color: int,
    noise_colors: _GenNoiseColors,
) -> bool:
    expected = {_GEN_EMPTY, frame_color, noise_colors[0], noise_colors[1]}
    if set(map(int, np.unique(grid))) != expected:
        return False
    background = grid[grid != frame_color]
    total = int(background.size)
    for color in (_GEN_EMPTY, noise_colors[0], noise_colors[1]):
        frac = float(np.count_nonzero(background == color)) / total
        if frac < _GEN_BALANCE_MIN or frac > _GEN_BALANCE_MAX:
            return False
    return True


def _gen_camouflage_target(
    inp: np.ndarray,
    base: np.ndarray,
    target: _GenBox,
    border: np.ndarray,
    noise_colors: _GenNoiseColors,
    rng: np.random.Generator,
) -> None:
    tr0, tc0, tr1, tc1 = target
    target_input = inp[tr0 : tr1 + 1, tc0 : tc1 + 1]
    target_base = base[tr0 : tr1 + 1, tc0 : tc1 + 1]
    target_input[border] = target_base[border]

    if not np.any(target_input[border] != _GEN_EMPTY):
        perimeter = np.argwhere(border)
        dr, dc = map(int, perimeter[int(rng.integers(len(perimeter)))])
        inp[tr0 + dr, tc0 + dc] = int(noise_colors[int(rng.integers(len(noise_colors)))])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(*_GEN_FRAME_SIZE_RANGE))
    width = int(rng.integers(*_GEN_FRAME_SIZE_RANGE))
    frame_color, noise_colors = _gen_palette(rng)
    source, target = _gen_place_pair(height, width, rng)
    base = _gen_noise_grid(noise_colors, rng)

    border = _gen_border_mask(height, width)
    inp = base.copy()
    _gen_stamp_frame(inp, source, frame_color, border)
    _gen_stamp_frame(inp, target, frame_color, border)
    tr0, tc0, tr1, tc1 = target
    _gen_camouflage_target(inp, base, target, border, noise_colors, rng)

    if not _gen_break_extra_windows(
        inp, height, width, source, target, frame_color, noise_colors, rng
    ):
        return None
    if not _gen_color_balance_ok(inp, frame_color, noise_colors):
        return None

    expected = inp.copy()
    expected[tr0 : tr1 + 1, tc0 : tc1 + 1][border] = frame_color
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 890034e9 input")
