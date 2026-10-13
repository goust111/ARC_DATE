from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator bounds.
PANEL_COUNT_RANGE = (2, 4)
MAX_COLORS_PER_PANEL = 3
MIN_TOTAL_COLORS = 3
GRID_HEIGHT_MIN = 8
GRID_HEIGHT_MAX = 18
PANEL_WIDTH_MIN = 5
PANEL_WIDTH_EXTRA_MAX = 4
SQUARE_SIDE_RANGE = (2, 5)
RECT_SIDE_RANGE = (2, 5)
SEPARATOR_WIDTH_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _panel_spans(g: np.ndarray) -> list[tuple[int, int]]:
    """Return maximal non-separator column spans [start, end)."""
    is_separator = np.all(g == GREY, axis=0)
    spans: list[tuple[int, int]] = []
    c = 0
    while c < g.shape[1]:
        if bool(is_separator[c]):
            c += 1
            continue
        start = c
        while c < g.shape[1] and not bool(is_separator[c]):
            c += 1
        spans.append((start, c))
    assert spans, "expected at least one panel"
    for c0, c1 in spans:
        assert not np.any(g[:, c0:c1] == GREY), "grey pixels should only appear in separator columns"
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge same-color pixels within each panel into one bounding box and fill
    only the colors whose merged box is square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    out = g.copy()
    for c0, c1 in _panel_spans(g):
        panel = g[:, c0:c1]
        colors = [int(color) for color in np.unique(panel) if int(color) not in {BLACK, GREY}]
        occupied = np.zeros(panel.shape, dtype=bool)

        for color in colors:
            rows, cols = np.where(panel == color)
            assert rows.size > 0
            r0, r1 = int(rows.min()), int(rows.max())
            cc0, cc1 = int(cols.min()), int(cols.max())
            h = r1 - r0 + 1
            w = cc1 - cc0 + 1
            if h != w:
                continue

            patch = panel[r0 : r1 + 1, cc0 : cc1 + 1]
            assert np.all((patch == BLACK) | (patch == color)), "square expansion must not cross other colors"
            region = occupied[r0 : r1 + 1, cc0 : cc1 + 1]
            assert not np.any(region), "square expansions must not overlap"
            region[:] = True
            out[r0 : r1 + 1, c0 + cc0 : c0 + cc1 + 1] = color

    return out


def _random_split(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _choose_panel_color_counts(panel_count: int) -> list[int]:
    total = random.randint(max(panel_count, MIN_TOTAL_COLORS), min(len(COLOR_POOL), panel_count * MAX_COLORS_PER_PANEL))
    counts = [1] * panel_count
    remaining = total - panel_count
    while remaining > 0:
        idx = random.choice([i for i, count in enumerate(counts) if count < MAX_COLORS_PER_PANEL])
        counts[idx] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _balanced_branches(total: int) -> list[bool]:
    square_count = total // 2
    if total % 2 and random.random() < 0.5:
        square_count += 1
    branches = [True] * square_count + [False] * (total - square_count)
    random.shuffle(branches)
    return branches


def _sample_bbox(square: bool) -> tuple[int, int]:
    if square:
        side = random.randint(*SQUARE_SIDE_RANGE)
        return side, side
    while True:
        h = random.randint(*RECT_SIDE_RANGE)
        w = random.randint(*RECT_SIDE_RANGE)
        if h != w:
            return h, w


def _sparse_mask(height: int, width: int, square_branch: bool) -> np.ndarray:
    """Draw a sparse motif whose merged bbox is exactly height x width."""
    area = height * width
    mask = np.zeros((height, width), dtype=bool)
    anchors = random.choice(
        [
            ((0, 0), (height - 1, width - 1)),
            ((0, width - 1), (height - 1, 0)),
        ]
    )
    for r, c in anchors:
        mask[r, c] = True

    if square_branch:
        max_target = min(area - 1, max(2, (2 * area) // 3))
    else:
        reserve = 2 if area >= 5 else 1
        max_target = min(area - reserve, max(2, area // 2))
    target = random.randint(2, max_target)

    choices = [(r, c) for r in range(height) for c in range(width) if not mask[r, c]]
    random.shuffle(choices)
    for r, c in choices:
        if int(mask.sum()) >= target:
            break
        mask[r, c] = True

    assert mask[0].any() and mask[-1].any()
    assert mask[:, 0].any() and mask[:, -1].any()
    assert not mask.all(), "input motif should stay sparse so the fill is visible"
    return mask


def _build_panel(specs: list[dict[str, int | bool]], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    panel = np.full((height, width), BLACK, dtype=int)
    expected = panel.copy()

    required_height = sum(int(spec["h"]) for spec in specs) + max(0, len(specs) - 1)
    slack = height - required_height
    gaps = _random_split(slack, len(specs) + 1)
    row = gaps[0]

    for i, spec in enumerate(specs):
        h = int(spec["h"])
        w = int(spec["w"])
        color = int(spec["color"])
        square_branch = bool(spec["square"])
        col = random.randint(0, width - w)
        mask = _sparse_mask(h, w, square_branch)

        patch = panel[row : row + h, col : col + w]
        patch[mask] = color
        if square_branch:
            expected[row : row + h, col : col + w] = color
        else:
            expected_patch = expected[row : row + h, col : col + w]
            expected_patch[mask] = color

        row += h
        if i + 1 < len(specs):
            row += 1 + gaps[i + 1]

    return panel, expected


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    panel_color_counts = _choose_panel_color_counts(panel_count)
    total_colors = sum(panel_color_counts)
    branches = _balanced_branches(total_colors)
    colors = random.sample(COLOR_POOL, total_colors)

    panel_specs: list[list[dict[str, int | bool]]] = []
    panel_widths: list[int] = []
    required_heights: list[int] = []
    cursor = 0
    for color_count in panel_color_counts:
        specs: list[dict[str, int | bool]] = []
        for _ in range(color_count):
            square_branch = branches[cursor]
            h, w = _sample_bbox(square_branch)
            specs.append({"color": colors[cursor], "square": square_branch, "h": h, "w": w})
            cursor += 1
        random.shuffle(specs)
        panel_specs.append(specs)
        required_heights.append(sum(int(spec["h"]) for spec in specs) + max(0, len(specs) - 1))
        min_width = max(PANEL_WIDTH_MIN, max(int(spec["w"]) for spec in specs))
        panel_widths.append(min_width + random.randint(0, PANEL_WIDTH_EXTRA_MAX))

    min_height = max(GRID_HEIGHT_MIN, max(required_heights))
    if min_height > GRID_HEIGHT_MAX:
        return None
    height = random.randint(min_height, GRID_HEIGHT_MAX)
    separator_widths = [random.choice(SEPARATOR_WIDTH_CHOICES) for _ in range(panel_count - 1)]
    width = sum(panel_widths) + sum(separator_widths)

    grid = np.full((height, width), BLACK, dtype=int)
    expected = grid.copy()
    col = 0
    square_seen = False
    nonsquare_seen = False

    for i, specs in enumerate(panel_specs):
        panel_w = panel_widths[i]
        panel, panel_expected = _build_panel(specs, height, panel_w)
        grid[:, col : col + panel_w] = panel
        expected[:, col : col + panel_w] = panel_expected
        square_seen |= any(bool(spec["square"]) for spec in specs)
        nonsquare_seen |= any(not bool(spec["square"]) for spec in specs)
        col += panel_w
        if i + 1 < panel_count:
            sep_w = separator_widths[i]
            grid[:, col : col + sep_w] = GREY
            expected[:, col : col + sep_w] = GREY
            col += sep_w

    if not (square_seen and nonsquare_seen):
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
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
