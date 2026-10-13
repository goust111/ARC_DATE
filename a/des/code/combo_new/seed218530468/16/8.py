from __future__ import annotations

import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 300
PATTERN_GAP = 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    ph, pw = map(int, pattern.shape)
    reps_r = (int(height) + ph - 1) // ph
    reps_c = (int(width) + pw - 1) // pw
    return np.tile(pattern, (reps_r, reps_c))[:height, :width]


def _extract_patterns(g: np.ndarray, bg: int) -> tuple[list[np.ndarray], set[int], int]:
    h, _w = g.shape
    boxes = []
    for color in np.unique(g):
        color = int(color)
        if color == int(bg):
            continue
        coords = np.argwhere(g == int(color))
        r0, c0, r1, c1 = _bbox(coords)
        boxes.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )

    groups: list[list[dict]] = []
    bottom_boxes = [box for box in boxes if int(box["r1"]) == int(h - 1)]
    for box in bottom_boxes:
        key = (int(box["r0"]), int(box["height"]), int(box["width"]))
        group = [item for item in bottom_boxes if (int(item["r0"]), int(item["height"]), int(item["width"])) == key]
        if len(group) < 2:
            continue

        group = sorted(group, key=lambda item: int(item["c0"]))
        ok = True
        for i, item in enumerate(group):
            r0 = int(item["r0"])
            r1 = int(item["r1"])
            c0 = int(item["c0"])
            c1 = int(item["c1"])
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.any((patch != int(bg)) & (patch != int(item["color"]))):
                ok = False
                break
            if i:
                prev = group[i - 1]
                gap_start = int(prev["c1"]) + 1
                gap_end = int(item["c0"]) - 1
                if gap_end - gap_start + 1 != PATTERN_GAP:
                    ok = False
                    break
                if np.any(g[r0 : r1 + 1, gap_start : gap_end + 1] != int(bg)):
                    ok = False
                    break
        if ok and group not in groups:
            groups.append(group)

    assert groups, "expected a bottom strip of equal-sized sample patterns"
    groups.sort(
        key=lambda items: (
            len(items),
            sum(int(item["width"]) for item in items),
            -int(items[0]["r0"]),
        ),
        reverse=True,
    )
    chosen = groups[0]
    sample_top = int(chosen[0]["r0"])
    sample_colors = {int(item["color"]) for item in chosen}
    patterns = [g[sample_top:h, int(item["c0"]) : int(item["c1"]) + 1].copy() for item in chosen]
    return patterns, sample_colors, sample_top


def _marker_rectangles(g: np.ndarray, bg: int, sample_colors: set[int], sample_top: int) -> list[dict]:
    rects = []
    for color in np.unique(g):
        color = int(color)
        if color == int(bg) or color in sample_colors:
            continue
        coords = np.argwhere(g == int(color))
        assert int(coords.shape[0]) == 2, "each marker color must appear exactly twice"
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 < int(sample_top), "markers must stay above the sample strip"
        corners = {(r0, c0), (r1, c1)}
        assert {tuple(map(int, cell)) for cell in coords} == corners, "marker pixels must be the top-left and bottom-right corners"
        rects.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    assert rects, "expected at least one marker pair"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marked rectangle with the sample pattern selected by its interior background count.

    Input structure:
    - The background is black.
    - Along the bottom is a left-to-right strip of equal-sized sample pattern boxes,
      each drawn in its own non-background color and separated by one background column.
    - Above the strip, every marker color appears exactly twice, at the top-left and
      bottom-right corners of an axis-aligned rectangle.

    Rule:
    - For each same-color marker pair, take its bounding rectangle.
    - Count how many background cells lie inside that rectangle; this count M selects
      the M-th sample pattern from the bottom strip, using 1-based indexing.
    - Tile the selected pattern over the whole rectangle from its top-left corner,
      crop the tiling to the rectangle bounds, and then restore the two marker pixels.
    - Leave the bottom sample strip unchanged.

    Key insight:
    - Marker colors only define the rectangle pairing; the true lookup key is the
      number of empty background cells inside the marked rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG
    patterns, sample_colors, sample_top = _extract_patterns(g, bg)
    rects = _marker_rectangles(g, bg, sample_colors, sample_top)

    out = g.copy()
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        marker_color = int(rect["color"])

        window = g[r0 : r1 + 1, c0 : c1 + 1]
        index = int(np.count_nonzero(window == int(bg)))
        assert 1 <= index <= len(patterns), "background count must select a sample pattern"

        tile = _tile_pattern(patterns[index - 1], r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = tile
        out[r0, c0] = marker_color
        out[r1, c1] = marker_color

    return out


def _factor_pairs(area: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    pairs = []
    for height in range(1, int(max_h) + 1):
        if area % height != 0:
            continue
        width = area // height
        if width <= int(max_w):
            pairs.append((int(height), int(width)))
    return pairs


def _random_mask(height: int, width: int, used: set[tuple[int, ...]]) -> np.ndarray | None:
    for _ in range(200):
        density = random.uniform(0.35, 0.8)
        mask = np.random.random((int(height), int(width))) < density
        mask[0, random.randrange(width)] = True
        mask[height - 1, random.randrange(width)] = True
        mask[random.randrange(height), 0] = True
        mask[random.randrange(height), width - 1] = True
        if int(mask.sum()) < 3:
            continue
        coords = np.argwhere(mask)
        if _bbox(coords) != (0, 0, int(height - 1), int(width - 1)):
            continue
        sig = tuple(int(v) for v in mask.reshape(-1))
        if sig in used:
            continue
        used.add(sig)
        return mask
    return None


def _build_patterns(height: int, width: int, colors: list[int]) -> list[np.ndarray] | None:
    used: set[tuple[int, ...]] = set()
    patterns = []
    for color in colors:
        mask = _random_mask(int(height), int(width), used)
        if mask is None:
            return None
        pattern = np.where(mask, int(color), BG).astype(int)
        patterns.append(pattern)
    return patterns


def _place_rectangles(
    grid: np.ndarray,
    top_height: int,
    marker_colors: list[int],
    max_index: int,
) -> list[dict] | None:
    h, w = grid.shape
    occupied = np.zeros((int(top_height), int(w)), dtype=bool)
    rects = []
    indices = [random.randint(1, int(max_index)) for _ in marker_colors]
    if len(indices) > 1 and len(set(indices)) == 1 and int(max_index) > 1:
        alternatives = [i for i in range(1, int(max_index) + 1) if i != indices[0]]
        indices[-1] = random.choice(alternatives)
    random.shuffle(indices)

    for marker_color, index in zip(marker_colors, indices):
        area = int(index) + 2
        pairs = _factor_pairs(area, int(top_height), int(w))
        if not pairs:
            return None
        dense = [pair for pair in pairs if pair[0] > 1 and pair[1] > 1]
        choices = dense if dense and random.random() < 0.7 else pairs

        placed = False
        for _ in range(200):
            height, width = random.choice(choices)
            r0 = random.randint(0, int(top_height) - int(height))
            c0 = random.randint(0, int(w) - int(width))

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(int(top_height), r0 + int(height) + 1)
            cc1 = min(int(w), c0 + int(width) + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue

            grid[r0, c0] = int(marker_color)
            grid[r0 + int(height) - 1, c0 + int(width) - 1] = int(marker_color)
            occupied[r0 : r0 + int(height), c0 : c0 + int(width)] = True
            rects.append(
                {
                    "color": int(marker_color),
                    "index": int(index),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r0 + int(height) - 1),
                    "c1": int(c0 + int(width) - 1),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return rects


def _render_expected(inp: np.ndarray, patterns: list[np.ndarray], rects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = int(rect["r1"])
        c1 = int(rect["c1"])
        marker_color = int(rect["color"])
        tile = _tile_pattern(patterns[int(rect["index"]) - 1], r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = tile
        out[r0, c0] = marker_color
        out[r1, c1] = marker_color
    return out


def _gen_attempt() -> np.ndarray | None:
    pattern_h = random.randint(2, 4)
    pattern_w = random.randint(2, 4)
    pattern_count = random.randint(3, 5)
    rect_count = random.randint(2, min(4, 9 - pattern_count))

    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    sample_width = left_margin + pattern_count * pattern_w + (pattern_count - 1) * PATTERN_GAP + right_margin

    gap_rows = random.randint(1, 3)
    top_height = random.randint(7, 18)
    grid_h = top_height + gap_rows + pattern_h
    if grid_h > 30:
        return None

    min_w = max(10, sample_width)
    grid_w = random.randint(min_w, min(30, max(min_w, min_w + 10)))

    colors = random.sample(list(range(1, 10)), pattern_count + rect_count)
    pattern_colors = colors[:pattern_count]
    marker_colors = colors[pattern_count:]

    patterns = _build_patterns(pattern_h, pattern_w, pattern_colors)
    if patterns is None:
        return None

    grid = np.zeros((int(grid_h), int(grid_w)), dtype=int)
    sample_top = int(grid_h - pattern_h)
    col = int(left_margin)
    for pattern in patterns:
        grid[sample_top:, col : col + pattern_w] = pattern
        col += pattern_w + PATTERN_GAP

    rects = _place_rectangles(grid, top_height, marker_colors, pattern_count)
    if rects is None:
        return None

    expected = _render_expected(grid, patterns, rects)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
