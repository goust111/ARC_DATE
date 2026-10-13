from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
SEPARATOR = GREY
FOREGROUND_COLORS = (BLUE, 3, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600
ANCHOR_MODES = ("aligned", "misaligned", "misaligned")

_GEN_CALL_COUNT = 0


def _background(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _split_panels(g: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    h, w = g.shape
    assert w % 2 == 1 and w >= 5, "expected two equal panels separated by one column"

    bg = _background(g)
    split_c = w // 2
    sep_col = g[:, split_c]
    assert np.unique(sep_col).size == 1, "separator must be a solid vertical line"
    sep = int(sep_col[0])
    assert sep != bg, "separator must differ from the background"

    left = g[:, :split_c]
    right = g[:, split_c + 1 :]
    assert left.shape == right.shape and left.size > 0
    assert not np.any(left == sep) and not np.any(right == sep), "separator color cannot occur inside panels"
    return bg, sep, left, right


def _single_anchor(panel: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(panel == RED)
    assert coords.shape == (1, 2), "each panel must contain exactly one red anchor"
    return int(coords[0, 0]), int(coords[0, 1])


def _panel_pattern(panel: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    colors = set(map(int, np.unique(panel))) - {int(bg), RED}
    assert len(colors) == 1, "each panel must have exactly one non-red foreground color"
    color = int(next(iter(colors)))
    mask = panel == color
    assert np.any(mask), "panel pattern must be non-empty"
    assert not bool(mask[_single_anchor(panel)]), "anchor cell cannot also be a pattern cell"
    return color, mask


def _registered_overlay(left: np.ndarray, right: np.ndarray, bg: int) -> np.ndarray:
    left_anchor = _single_anchor(left)
    right_anchor = _single_anchor(right)
    left_color, left_mask = _panel_pattern(left, bg)
    right_color, right_mask = _panel_pattern(right, bg)

    out = np.full(left.shape, int(bg), dtype=int)
    out[left_mask] = int(left_color)

    dr = left_anchor[0] - right_anchor[0]
    dc = left_anchor[1] - right_anchor[1]
    h, w = left.shape
    for r, c in np.argwhere(right_mask):
        rr = int(r) + dr
        cc = int(c) + dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        out[rr, cc] = YELLOW if bool(left_mask[rr, cc]) else int(right_color)

    out[left_anchor] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Register two anchor-marked panels and overlay their scattered patterns.

    Input structure:
    - A solid vertical separator divides the input into two equal-size side-by-side panels.
    - Each panel has a black background, exactly one red anchor pixel, and one non-red
      foreground color forming a sparse scattered pattern.
    - The two patterns are read in their own panel coordinates; their anchors may or may
      not occupy the same raw row and column.

    Rule:
    - Translate the right panel so its red anchor lands on the left panel's red anchor.
    - Return one panel-sized grid: cells occupied by both registered non-red patterns are
      yellow, cells occupied by only one side keep that side's color, and all other cells
      are background.
    - The registered anchor position remains a single red pixel.

    Key insight:
    - The red anchor is the unique correspondence between the two coordinate systems; raw
      cell-by-cell overlay is valid only in the special case where the anchors already align.
    """
    g = np.asarray(grid, dtype=int)
    bg, _sep, left, right = _split_panels(g)
    return _registered_overlay(left, right, bg)


# --------------- generator ---------------


def _panel_from_offsets(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    offsets: set[tuple[int, int]],
    color: int,
) -> np.ndarray:
    panel = np.full(shape, BLACK, dtype=int)
    ar, ac = anchor
    for dr, dc in offsets:
        panel[ar + int(dr), ac + int(dc)] = int(color)
    panel[ar, ac] = RED
    return panel


def _expected_from_specs(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    left_offsets: set[tuple[int, int]],
    right_offsets: set[tuple[int, int]],
    left_color: int,
    right_color: int,
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    ar, ac = anchor
    for dr, dc in left_offsets:
        out[ar + int(dr), ac + int(dc)] = int(left_color)
    for dr, dc in right_offsets:
        r = ar + int(dr)
        c = ac + int(dc)
        out[r, c] = YELLOW if (int(dr), int(dc)) in left_offsets else int(right_color)
    out[ar, ac] = RED
    return out


def _compose_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = left.shape
    grid = np.full((h, 2 * w + 1), BLACK, dtype=int)
    grid[:, w] = SEPARATOR
    grid[:, :w] = left
    grid[:, w + 1 :] = right
    return grid


def _raw_overlay(left: np.ndarray, right: np.ndarray, bg: int) -> np.ndarray:
    left_color, left_mask = _panel_pattern(left, bg)
    right_color, right_mask = _panel_pattern(right, bg)
    out = np.full(left.shape, int(bg), dtype=int)
    out[left_mask] = int(left_color)
    for r, c in np.argwhere(right_mask):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = YELLOW if bool(left_mask[rr, cc]) else int(right_color)
    for r, c in np.argwhere((left == RED) | (right == RED)):
        out[int(r), int(c)] = RED
    return out


def _valid_offsets(
    shape: tuple[int, int],
    left_anchor: tuple[int, int],
    right_anchor: tuple[int, int],
    radius: int,
) -> list[tuple[int, int]]:
    h, w = shape
    lar, lac = left_anchor
    rar, rac = right_anchor
    offsets: list[tuple[int, int]] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) < 2:
                continue
            if not (0 <= lar + dr < h and 0 <= lac + dc < w):
                continue
            if not (0 <= rar + dr < h and 0 <= rac + dc < w):
                continue
            offsets.append((int(dr), int(dc)))
    return offsets


def _sample_spaced_offsets(
    rng: np.random.Generator,
    candidates: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    items = list(candidates)
    rng.shuffle(items)
    chosen: list[tuple[int, int]] = []
    for offset in items:
        if all(max(abs(offset[0] - old[0]), abs(offset[1] - old[1])) >= 2 for old in chosen):
            chosen.append(offset)
            if len(chosen) == count:
                return chosen
    return None


def _choose_anchors(
    rng: np.random.Generator, shape: tuple[int, int], mode: str
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    rows = list(range(2, h - 2))
    cols = list(range(2, w - 2))
    if not rows or not cols:
        return None

    left = (int(rng.choice(rows)), int(rng.choice(cols)))
    if mode == "aligned":
        return left, left

    options = [
        (r, c)
        for r in rows
        for c in cols
        if (r, c) != left and abs(r - left[0]) + abs(c - left[1]) >= 2
    ]
    if not options:
        return None
    right = options[int(rng.integers(len(options)))]
    return left, (int(right[0]), int(right[1]))


def _semantic_check(inp: np.ndarray, expected: np.ndarray, mode: str) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    try:
        bg, _sep, left, right = _split_panels(inp)
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    colors = set(map(int, np.unique(expected)))
    if not {BLACK, RED, YELLOW}.issubset(colors):
        return False
    left_color, _left_mask = _panel_pattern(left, bg)
    right_color, _right_mask = _panel_pattern(right, bg)
    if left_color == right_color or left_color == YELLOW or right_color == YELLOW:
        return False
    if int(np.count_nonzero(expected == YELLOW)) < 2:
        return False
    if int(np.count_nonzero(expected == left_color)) < 2:
        return False
    if int(np.count_nonzero(expected == right_color)) < 2:
        return False

    raw = _raw_overlay(left, right, bg)
    if mode == "misaligned" and np.array_equal(raw, expected):
        return False
    if mode == "aligned" and not np.array_equal(raw, expected):
        return False

    no_overlap_recolor = expected.copy()
    no_overlap_recolor[no_overlap_recolor == YELLOW] = left_color
    if np.array_equal(no_overlap_recolor, expected):
        return False
    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(9, 19))
    panel_w = int(rng.integers(9, 15))
    if 2 * panel_w + 1 > 30:
        return None

    anchors = _choose_anchors(rng, (height, panel_w), mode)
    if anchors is None:
        return None
    left_anchor, right_anchor = anchors

    radius = int(rng.integers(3, 6))
    candidates = _valid_offsets((height, panel_w), left_anchor, right_anchor, radius)
    common_count = int(rng.integers(2, 5))
    left_extra_count = int(rng.integers(2, 5))
    right_extra_count = int(rng.integers(2, 5))
    total = common_count + left_extra_count + right_extra_count
    offsets = _sample_spaced_offsets(rng, candidates, total)
    if offsets is None:
        return None

    common = set(offsets[:common_count])
    left_extra = set(offsets[common_count : common_count + left_extra_count])
    right_extra = set(offsets[common_count + left_extra_count :])
    left_offsets = common | left_extra
    right_offsets = common | right_extra

    left_color, right_color = [int(c) for c in rng.choice(FOREGROUND_COLORS, size=2, replace=False)]
    left = _panel_from_offsets((height, panel_w), left_anchor, left_offsets, left_color)
    right = _panel_from_offsets((height, panel_w), right_anchor, right_offsets, right_color)
    inp = _compose_input(left, right)
    expected = _expected_from_specs(
        (height, panel_w), left_anchor, left_offsets, right_offsets, left_color, right_color
    )
    return inp if _semantic_check(inp, expected, mode) else None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = ANCHOR_MODES[_GEN_CALL_COUNT % len(ANCHOR_MODES)]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid anchor-registration overlay input")
