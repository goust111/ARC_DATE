from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
COLOR_POOL = tuple(range(10))
NON_GREY_COLORS = tuple(c for c in COLOR_POOL if c != GREY)

GRID_WIDTH_MAX = 30
STAMP_COUNT_RANGE = (2, 4)
STAMP_HEIGHT_RANGE = (4, 8)
STAMP_WIDTH_MIN = 4
STAMP_WIDTH_CAP = 7
LEFT_PAD_RANGE = (0, 2)
RIGHT_PAD_RANGE = (0, 2)
GAP_RANGE = (1, 2)
GEN_RETRIES = 400

PATTERN_MODES = ("rect_hole", "stripe_holes", "checker_holes", "scatter_holes")
REPEATED_SEQUENCE_PROB = 0.5


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(np.argmax(counts))
    return int(values[best])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return r0, c0, r1, c1


def _find_separator_column(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg:
            candidates.append((int(grid[0, c]), c))

    grey_cols = [c for color, c in candidates if color == GREY]
    if len(grey_cols) == 1:
        return grey_cols[0]

    assert len(candidates) == 1, "expected one uniform separator column"
    return candidates[0][1]


def _extract_stamps(stamp_area: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    colors = [int(c) for c in np.unique(stamp_area) if int(c) != bg]
    assert colors, "expected at least one stamp color"

    stamps: dict[int, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for color in colors:
        coords = np.argwhere(stamp_area == color)
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        patch = stamp_area[r0 : r1 + 1, c0 : c1 + 1]
        assert set(map(int, np.unique(patch))).issubset({bg, color}), "stamps must be monochrome"
        assert np.all(patch[0, :] == color)
        assert np.all(patch[-1, :] == color)
        assert np.all(patch[:, 0] == color)
        assert np.all(patch[:, -1] == color)
        mask = patch == color
        assert not mask.all(), "each stamp needs transparent holes"
        if shape is None:
            shape = mask.shape
        assert mask.shape == shape, "all stamps must share one size"
        stamps[color] = mask

    assert shape is not None
    return stamps, shape


def _stamp_sequence(order_row: np.ndarray, bg: int) -> list[int]:
    sequence = [int(v) for v in order_row if int(v) != bg]
    assert sequence, "expected a non-empty order bar"
    return sequence


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the colored stamp rectangles on the left, then simulate transparent stamping
    in the order shown by the left-half bottom row. The output is the final canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3

    bg = _most_common_color(g)
    sep_col = _find_separator_column(g, bg)
    assert 0 < sep_col < g.shape[1] - 1

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape[1] > 0 and right.shape[1] > 0

    sequence = _stamp_sequence(left[-1], bg)
    stamps, stamp_shape = _extract_stamps(left[:-1], bg)
    assert set(sequence).issubset(stamps), "order bar colors must map to stamps"

    canvas_h, canvas_w = stamp_shape
    assert right.shape == (canvas_h + 1, canvas_w), "right half must be the blank canvas plus the order-row margin"
    assert np.all(right[:-1] == bg), "canvas must start blank"

    out = np.full((canvas_h, canvas_w), bg, dtype=int)
    for color in sequence:
        mask = stamps[color]
        out[mask] = color
    return out


def _rect_hole_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    hole_h = random.randint(1, h - 2)
    hole_w = random.randint(1, w - 2)
    r0 = random.randint(1, h - 1 - hole_h)
    c0 = random.randint(1, w - 1 - hole_w)
    mask[r0 : r0 + hole_h, c0 : c0 + hole_w] = False
    return mask


def _stripe_hole_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    interior = mask[1:-1, 1:-1]
    horizontal = random.choice([True, False])
    phase = random.randint(0, 1)
    if horizontal:
        interior[phase::2, :] = False
    else:
        interior[:, phase::2] = False
    if not interior.any():
        interior[random.randrange(interior.shape[0]), random.randrange(interior.shape[1])] = True
    return mask


def _checker_hole_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    interior = mask[1:-1, 1:-1]
    rr, cc = np.indices(interior.shape)
    phase = random.randint(0, 1)
    interior[(rr + cc + phase) % 2 == 0] = False
    if not interior.any():
        interior[0, 0] = True
    return mask


def _scatter_hole_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    interior = mask[1:-1, 1:-1]
    total = interior.size
    holes = random.randint(max(1, total // 4), max(1, (2 * total) // 3))
    cells = random.sample(range(total), holes)
    rows, cols = np.unravel_index(cells, interior.shape)
    interior[rows, cols] = False
    if not interior.any():
        interior[random.randrange(interior.shape[0]), random.randrange(interior.shape[1])] = True
    return mask


def _make_stamp_mask(h: int, w: int, mode: str) -> np.ndarray:
    if mode == "rect_hole":
        mask = _rect_hole_mask(h, w)
    elif mode == "stripe_holes":
        mask = _stripe_hole_mask(h, w)
    elif mode == "checker_holes":
        mask = _checker_hole_mask(h, w)
    elif mode == "scatter_holes":
        mask = _scatter_hole_mask(h, w)
    else:
        raise AssertionError("unknown stamp pattern mode")

    assert np.all(mask[0, :]) and np.all(mask[-1, :]) and np.all(mask[:, 0]) and np.all(mask[:, -1])
    assert np.any(~mask[1:-1, 1:-1]), "holes must stay inside the rectangle"
    return mask


def _render_output(stamps: dict[int, np.ndarray], sequence: list[int], bg: int) -> np.ndarray:
    shape = next(iter(stamps.values())).shape
    out = np.full(shape, bg, dtype=int)
    for color in sequence:
        out[stamps[color]] = color
    return out


def _single_stamp_canvas(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    return np.where(mask, color, bg)


def _sequence_has_required_layering(
    stamps: dict[int, np.ndarray], sequence: list[int], output: np.ndarray, bg: int
) -> bool:
    if len(sequence) < 2:
        return False

    filled = np.zeros_like(output, dtype=bool)
    overwrite_seen = False
    for color in sequence:
        mask = stamps[color]
        if np.any(filled & mask):
            overwrite_seen = True
        filled |= mask
    if not overwrite_seen:
        return False

    visible_colors = {int(v) for v in np.unique(output) if int(v) != bg}
    if len(visible_colors) < 2:
        return False

    last_color = sequence[-1]
    if not np.any((output != bg) & (output != last_color) & ~stamps[last_color]):
        return False

    for color, mask in stamps.items():
        if np.array_equal(output, _single_stamp_canvas(mask, color, bg)):
            return False
    return True


def _unique_dominant_bg(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    bg_count = counts[bg]
    return all(color == bg or bg_count > count for color, count in counts.items())


def _sample_layout() -> tuple[int, int, int, int, list[int]] | None:
    n_stamps = random.randint(*STAMP_COUNT_RANGE)
    left_pad = random.randint(*LEFT_PAD_RANGE)
    right_pad = random.randint(*RIGHT_PAD_RANGE)
    gaps = [random.randint(*GAP_RANGE) for _ in range(n_stamps - 1)]
    extra = left_pad + right_pad + sum(gaps) + 1
    max_w = min(STAMP_WIDTH_CAP, (GRID_WIDTH_MAX - extra) // (n_stamps + 1))
    if max_w < STAMP_WIDTH_MIN:
        return None
    stamp_w = random.randint(STAMP_WIDTH_MIN, max_w)
    stamp_h = random.randint(*STAMP_HEIGHT_RANGE)
    return n_stamps, stamp_h, stamp_w, left_pad, gaps + [right_pad]


def _sample_sequence(colors: list[int]) -> list[int]:
    base = colors[:]
    random.shuffle(base)
    if random.random() >= REPEATED_SEQUENCE_PROB:
        return base

    extra_count = random.randint(1, len(colors))
    extras = random.choices(colors, k=extra_count)
    seq = base + extras
    random.shuffle(seq)
    return seq


def _build_input(
    bg: int,
    stamp_colors: list[int],
    stamp_masks: dict[int, np.ndarray],
    sequence: list[int],
    left_pad: int,
    gaps_with_right_pad: list[int],
) -> np.ndarray:
    stamp_h, stamp_w = next(iter(stamp_masks.values())).shape
    right_pad = gaps_with_right_pad[-1]
    gaps = gaps_with_right_pad[:-1]
    left_w = left_pad + len(stamp_colors) * stamp_w + sum(gaps) + right_pad
    height = stamp_h + 1
    width = left_w + 1 + stamp_w

    grid = np.full((height, width), bg, dtype=int)
    sep_col = left_w
    grid[:, sep_col] = GREY

    c = left_pad
    for idx, color in enumerate(stamp_colors):
        mask = stamp_masks[color]
        patch = np.where(mask, color, bg)
        grid[:stamp_h, c : c + stamp_w] = patch
        c += stamp_w
        if idx < len(gaps):
            c += gaps[idx]

    start = random.randint(0, left_w - len(sequence))
    for offset, color in enumerate(sequence):
        grid[-1, start + offset] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    n_stamps, stamp_h, stamp_w, left_pad, gaps_with_right_pad = layout

    bg = random.choice(NON_GREY_COLORS)
    stamp_colors = random.sample([c for c in NON_GREY_COLORS if c != bg], n_stamps)

    modes = [random.choice(PATTERN_MODES) for _ in range(n_stamps)]
    stamp_masks = {
        color: _make_stamp_mask(stamp_h, stamp_w, mode)
        for color, mode in zip(stamp_colors, modes)
    }
    if len({tuple(mask.ravel().tolist()) for mask in stamp_masks.values()}) < 2:
        return None

    sequence = _sample_sequence(stamp_colors)
    output = _render_output(stamp_masks, sequence, bg)
    if not _sequence_has_required_layering(stamp_masks, sequence, output, bg):
        return None

    grid = _build_input(bg, stamp_colors, stamp_masks, sequence, left_pad, gaps_with_right_pad)
    if not _unique_dominant_bg(grid, bg):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
