from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_MAX_ATTEMPTS = 400
GEN_CANVAS_SIDES = (8, 10)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _find_canvas(grid: np.ndarray) -> tuple[int, int, int]:
    canvas_mask = grid == GREEN
    comps = _connected_components(canvas_mask)
    assert len(comps) == 1, "expected exactly one green canvas"
    r0, c0, r1, c1 = _bbox(canvas_mask)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w and side_h % 2 == 0, "canvas must be an even square"
    assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN)), "canvas must be solid"
    return r0, c0, side_h


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    bar_mask = np.isin(grid, [BLUE, RED])
    bars: list[dict[str, int]] = []
    for coords in _connected_components(bar_mask):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        color_values = {int(grid[r, c]) for r, c in coords}
        assert len(color_values) == 1, "each bar must be monochromatic"
        color = color_values.pop()
        assert color in (BLUE, RED)
        assert c0 == c1, "bars must be one column wide"
        assert int(coords.shape[0]) == r1 - r0 + 1, "bars must be solid vertical segments"
        bars.append({"color": color, "col": c0, "height": r1 - r0 + 1, "bottom": r1})
    assert 3 <= len(bars) <= 5, "expected 3 to 5 bars"
    assert len({bar["height"] for bar in bars}) == len(bars), "bar heights must be distinct"
    return bars


def _build_canvas(side: int, heights_desc: list[int], majority_color: int) -> np.ndarray:
    quadrant_side = side // 2
    assert len(heights_desc) <= quadrant_side
    assert all(0 < h <= quadrant_side for h in heights_desc)

    carved = np.full((quadrant_side, quadrant_side), GREEN, dtype=int)
    for idx, height in enumerate(heights_desc):
        carved[:height, idx] = BLACK

    if majority_color == BLUE:
        top = np.concatenate([carved, np.fliplr(carved)], axis=1)
    else:
        assert majority_color == RED
        top = np.concatenate([np.fliplr(carved), carved], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)


def _render_output(shape: tuple[int, int], canvas_origin: tuple[int, int], side: int, heights_desc: list[int], majority_color: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, c0 = canvas_origin
    out[r0 : r0 + side, c0 : c0 + side] = _build_canvas(side, heights_desc, majority_color)
    return out


def _parsed_semantics(grid: np.ndarray) -> tuple[tuple[int, int], int, list[int], int, list[dict[str, int]]]:
    canvas_r0, canvas_c0, side = _find_canvas(grid)
    bars = _extract_bars(grid)
    heights_desc = sorted((bar["height"] for bar in bars), reverse=True)
    blue_count = sum(bar["color"] == BLUE for bar in bars)
    red_count = len(bars) - blue_count
    assert blue_count != red_count, "bar colors must have a strict majority"
    majority_color = BLUE if blue_count > red_count else RED
    return (canvas_r0, canvas_c0), side, heights_desc, majority_color, bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve a mirrored green canvas using the height order of top bars and the majority bar color as the quadrant vote.

    Input structure:
    - The grid has a black background with several one-cell-wide blue/red vertical bars in the upper half.
    - The lower half contains one solid green even-sized square canvas.
    - Bar heights are distinct; bar colors encode a blue-vs-red majority vote.

    Rule:
    - Read every blue/red bar, sort their heights from tallest to shortest, and keep only that descending height sequence.
    - Count bar colors: blue majority selects the canvas upper-left quadrant, red majority selects the upper-right quadrant.
    - In the selected upper quadrant, erase from the top of column 1, 2, ... by the sorted heights, then mirror that carved quadrant across the vertical center and then the horizontal center to refill the whole canvas.
    - Remove the entire bar region in the output so only the transformed canvas remains on black.

    Key insight:
    - The same bar set carries two independent signals: heights decide how deep each carved column is, while color majority decides which upper quadrant is the seed before four-quadrant mirroring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    canvas_origin, side, heights_desc, majority_color, _bars = _parsed_semantics(g)
    return _render_output(g.shape, canvas_origin, side, heights_desc, majority_color)


def _sample_columns(width: int, count: int) -> list[int] | None:
    choices = list(range(1, width - 1))
    random.shuffle(choices)
    cols: list[int] = []
    for col in choices:
        if all(abs(col - other) >= 2 for other in cols):
            cols.append(col)
            if len(cols) == count:
                return sorted(cols)
    return None


def _majority_counts(count: int) -> tuple[int, int]:
    if count == 3:
        return 2, 1
    if count == 4:
        return 3, 1
    raise AssertionError("generator uses only 3 or 4 bars")


def _alternative_output(shape: tuple[int, int], canvas_origin: tuple[int, int], side: int, bars: list[dict[str, int]], mode: str) -> np.ndarray:
    bars_by_col = sorted(bars, key=lambda bar: bar["col"])
    if mode == "lr_order":
        heights = [bar["height"] for bar in bars_by_col]
        majority_color = BLUE if sum(bar["color"] == BLUE for bar in bars) > sum(bar["color"] == RED for bar in bars) else RED
    elif mode == "leftmost_color":
        heights = sorted((bar["height"] for bar in bars), reverse=True)
        majority_color = bars_by_col[0]["color"]
    elif mode == "tallest_color":
        heights = sorted((bar["height"] for bar in bars), reverse=True)
        tallest = max(bars, key=lambda bar: bar["height"])
        majority_color = tallest["color"]
    elif mode == "opposite_quadrant":
        heights = sorted((bar["height"] for bar in bars), reverse=True)
        true_majority = BLUE if sum(bar["color"] == BLUE for bar in bars) > sum(bar["color"] == RED for bar in bars) else RED
        majority_color = RED if true_majority == BLUE else BLUE
    else:
        raise ValueError(f"unknown mode: {mode}")
    return _render_output(shape, canvas_origin, side, heights, majority_color)


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        canvas_origin, side, heights_desc, majority_color, bars = _parsed_semantics(inp)
    except AssertionError:
        return False

    produced = main(inp)
    if not np.array_equal(produced, expected):
        return False

    canvas_r0, canvas_c0 = canvas_origin
    if np.any(produced == BLUE) or np.any(produced == RED):
        return False
    if not np.array_equal(produced == GREEN, expected == GREEN):
        return False
    if not np.all(produced[:canvas_r0, :] == BLACK):
        return False
    if not np.all(produced[canvas_r0 + side :, :] == BLACK):
        return False
    if not np.all(produced[:, :canvas_c0][produced[:, :canvas_c0] != BLACK] == GREEN):
        return False

    bars_by_col = sorted(bars, key=lambda bar: bar["col"])
    lr_heights = [bar["height"] for bar in bars_by_col]
    if lr_heights == heights_desc or lr_heights == list(reversed(heights_desc)):
        return False

    if np.array_equal(_alternative_output(inp.shape, canvas_origin, side, bars, "lr_order"), expected):
        return False

    color_alts = 0
    for mode in ("leftmost_color", "tallest_color", "opposite_quadrant"):
        alt = _alternative_output(inp.shape, canvas_origin, side, bars, mode)
        if np.array_equal(alt, expected):
            if mode == "opposite_quadrant":
                return False
        else:
            color_alts += 1
    if color_alts == 0:
        return False

    rebuilt = _render_output(inp.shape, canvas_origin, side, heights_desc, majority_color)
    return np.array_equal(rebuilt, expected)


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(GEN_CANVAS_SIDES)
    quadrant_side = side // 2
    possible_heights = list(range(2, quadrant_side + 1))
    bar_count = 3 if side == 8 else random.choice([3, 4])
    if bar_count > len(possible_heights):
        return None

    heights_desc = sorted(random.sample(possible_heights, bar_count), reverse=True)
    shuffled_heights = heights_desc[:]
    random.shuffle(shuffled_heights)
    if shuffled_heights == heights_desc or shuffled_heights == list(reversed(heights_desc)):
        return None

    majority_color = random.choice([BLUE, RED])
    majority_count, minority_count = _majority_counts(bar_count)
    colors = [majority_color] * majority_count + ([RED if majority_color == BLUE else BLUE] * minority_count)
    random.shuffle(colors)

    height = random.randint(2 * side, min(26, 2 * side + 4))
    width = random.randint(max(side + 4, 2 * bar_count + 8), min(26, side + 10))

    canvas_r0 = random.randint((height + 1) // 2, height - side)
    canvas_c0 = random.randint(1, width - side - 1)
    max_bar_height = max(heights_desc)
    bar_base_lo = max_bar_height
    bar_base_hi = min(height // 2 - 1, canvas_r0 - 2)
    if bar_base_lo > bar_base_hi:
        return None
    bar_base = random.randint(bar_base_lo, bar_base_hi)

    cols = _sample_columns(width, bar_count)
    if cols is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    inp[canvas_r0 : canvas_r0 + side, canvas_c0 : canvas_c0 + side] = GREEN

    bars: list[dict[str, int]] = []
    for col, height_i, color in zip(cols, shuffled_heights, colors):
        inp[bar_base - height_i + 1 : bar_base + 1, col] = color
        bars.append({"color": int(color), "col": int(col), "height": int(height_i), "bottom": int(bar_base)})

    expected = _render_output(inp.shape, (canvas_r0, canvas_c0), side, heights_desc, majority_color)
    if not _semantic_check(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
