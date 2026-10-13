from __future__ import annotations

from itertools import product
from pathlib import Path
import random
import sys

import numpy as np


def _add_arc_utils_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.append(utils_dir)
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_add_arc_utils_to_path()

from arc_utils import Color, bbox  # noqa: E402


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
Marker = tuple[int, int, int]


PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

FRAME_HEIGHT_RANGE = (8, 13)
REGION_WIDTH_RANGE = (2, 4)
CANVAS_EXTRA_RANGE = (4, 10)
MAX_GENERATION_ATTEMPTS = 500


def _find_frame_box(grid: np.ndarray) -> BBox:
    frame_box = bbox(grid == Color.GREY)
    assert frame_box is not None, "expected a grey frame"
    r0, c0, r1, c1 = frame_box
    assert r1 - r0 >= 3 and c1 - c0 >= 4, "frame must enclose a non-empty interior"

    assert np.all(grid[r0 : r1 + 1, c0] == Color.GREY), "left edge must stay grey"
    assert np.all(grid[r0 : r1 + 1, c1] == Color.GREY), "right edge must stay grey"
    assert np.all(grid[r1, c0 : c1 + 1] == Color.GREY), "bottom edge must stay grey"
    assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == Color.GREY), "frame interior must not contain grey"
    return frame_box


def _top_split_points(grid: np.ndarray, frame_box: BBox) -> tuple[list[int], list[int]]:
    r0, c0, _r1, c1 = frame_box
    split_cols: list[int] = []
    split_colors: list[int] = []

    assert int(grid[r0, c0]) == Color.GREY and int(grid[r0, c1]) == Color.GREY, "frame corners must be grey"
    for c in range(c0 + 1, c1):
        value = int(grid[r0, c])
        assert value != Color.BLACK, "top edge cannot contain black gaps"
        if value == Color.GREY:
            continue
        split_cols.append(c)
        split_colors.append(value)

    assert 2 <= len(split_cols) <= 4, "expected 2 to 4 top split points"
    assert len(set(split_cols)) == len(split_cols), "split columns must be unique"
    assert len(set(split_colors)) == len(split_colors), "split colors must be unique"
    return split_cols, split_colors


def _region_spans(frame_box: BBox, split_cols: list[int]) -> list[tuple[int, int]]:
    _r0, c0, _r1, c1 = frame_box
    bounds = [c0, *split_cols, c1]
    return [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


def _marker_palette(grid: np.ndarray, split_colors: list[int]) -> list[int]:
    visible = {
        int(value)
        for value in np.unique(grid)
        if int(value) not in {Color.BLACK, Color.GREY}
    }
    return sorted(visible - set(split_colors))


def _inside_frame(frame_box: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = frame_box
    return r0 < r < r1 and c0 < c < c1


def _region_markers(
    grid: np.ndarray,
    frame_box: BBox,
    regions: list[tuple[int, int]],
    marker_colors: list[int],
) -> list[Marker | None]:
    r0, _c0, r1, _c1 = frame_box
    if not marker_colors:
        return [None] * len(regions)

    marker_mask = np.isin(grid, marker_colors)
    per_region: list[Marker | None] = []
    for left, right in regions:
        found: list[Marker] = []
        for local_r, local_c in np.argwhere(marker_mask[r0 + 1 : r1, left : right + 1]):
            rr = int(local_r) + r0 + 1
            cc = int(local_c) + left
            found.append((rr, cc, int(grid[rr, cc])))
        assert len(found) <= 1, "each region may contain at most one active marker"
        per_region.append(found[0] if found else None)
    return per_region


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame_box = _find_frame_box(g)
    split_cols, split_colors = _top_split_points(g, frame_box)
    regions = _region_spans(frame_box, split_cols)
    marker_colors = _marker_palette(g, split_colors)
    active_markers = _region_markers(g, frame_box, regions, marker_colors)

    assert any(marker is not None for marker in active_markers), "expected at least one active marker"

    out = g.copy()
    if marker_colors:
        out[np.isin(out, marker_colors)] = Color.BLACK

    r0, _c0, r1, _c1 = frame_box
    for c, color in zip(split_cols, split_colors):
        out[r0 : r1 + 1, c] = color

    for marker, (left, right) in zip(active_markers, regions):
        if marker is None:
            continue
        _mr, _mc, color = marker
        region = out[r0 + 1 : r1, left : right + 1]
        region[region == Color.BLACK] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame subregion from the marker inside it, where top-edge colored pixels cast vertical partition rays.

    Input structure:
    - The grid has a black background and one grey hollow rectangle frame.
    - Several non-grey, non-black pixels are embedded in the frame's top edge; these are the ray split points.
    - Single-pixel marker colors appear both inside and outside the frame, and the outside ones are distractors.

    Rule:
    - Find the grey frame's bounding box and read the colored pixels on its top edge.
    - From each top-edge colored pixel, draw a vertical ray of that color straight down through the frame to the bottom edge.
    - Keep only the marker pixels strictly inside the frame interior; erase same-colored markers outside the frame.
    - The ray columns partition the frame interior into vertical subregions; if a subregion contains one kept marker, recolor every black cell in that subregion to the marker's color without overwriting the rays.

    Key insight:
    - The top-edge colors matter for their positions, not for matching marker colors: they create the ray barriers that decide which marker controls which fill region.
    """
    return _oracle_output(grid)


def _occupancy_patterns(n_regions: int) -> list[tuple[int, ...]]:
    patterns = []
    for pattern in product((0, 1), repeat=n_regions):
        if 0 < sum(pattern) < n_regions and all(pattern[i] or pattern[i + 1] for i in range(n_regions - 1)):
            patterns.append(pattern)
    return patterns


def _sample_region_widths(n_regions: int) -> list[int]:
    for _ in range(100):
        widths = [random.randint(*REGION_WIDTH_RANGE) for _ in range(n_regions)]
        if len(set(widths)) > 1:
            return widths
    return [random.randint(*REGION_WIDTH_RANGE) for _ in range(n_regions)]


def _build_frame_columns(c0: int, region_widths: list[int]) -> tuple[int, list[int], list[tuple[int, int]]]:
    split_cols: list[int] = []
    regions: list[tuple[int, int]] = []
    cursor = c0 + 1
    for index, width in enumerate(region_widths):
        regions.append((cursor, cursor + width - 1))
        cursor += width
        if index != len(region_widths) - 1:
            split_cols.append(cursor)
            cursor += 1
    c1 = cursor
    return c1, split_cols, regions


def _draw_input_grid(
    canvas_shape: tuple[int, int],
    frame_box: BBox,
    split_cols: list[int],
    split_colors: list[int],
    active_by_region: list[Marker | None],
    distractors: list[Marker],
) -> np.ndarray:
    h, w = canvas_shape
    grid = np.full((h, w), Color.BLACK, dtype=int)
    r0, c0, r1, c1 = frame_box

    grid[r0, c0 : c1 + 1] = Color.GREY
    grid[r1, c0 : c1 + 1] = Color.GREY
    grid[r0 : r1 + 1, c0] = Color.GREY
    grid[r0 : r1 + 1, c1] = Color.GREY
    for c, color in zip(split_cols, split_colors):
        grid[r0, c] = color

    for marker in active_by_region:
        if marker is None:
            continue
        r, c, color = marker
        grid[r, c] = color

    for r, c, color in distractors:
        grid[r, c] = color

    return grid


def _outside_candidates(shape: tuple[int, int], frame_box: BBox) -> tuple[list[Coord], list[Coord]]:
    h, w = shape
    r0, c0, r1, c1 = frame_box
    strict: list[Coord] = []
    loose: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            loose.append((r, c))
            if not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1):
                strict.append((r, c))
    return strict, loose


def _pick_spaced_points(candidates: list[Coord], count: int) -> list[Coord] | None:
    random.shuffle(candidates)
    chosen: list[Coord] = []
    for r, c in candidates:
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            return chosen
    return None


def _states_from_regions(active_by_region: list[Marker | None]) -> list[int]:
    return [0 if marker is None else int(marker[2]) for marker in active_by_region]


def _validate_case(grid: np.ndarray, active_by_region: list[Marker | None], distractors: list[Marker]) -> None:
    frame_box = _find_frame_box(grid)
    split_cols, split_colors = _top_split_points(grid, frame_box)
    regions = _region_spans(frame_box, split_cols)
    marker_colors = _marker_palette(grid, split_colors)
    inferred = _region_markers(grid, frame_box, regions, marker_colors)

    assert inferred == active_by_region, "generated active markers must be uniquely recoverable"
    assert any(marker is None for marker in active_by_region), "at least one subregion must stay empty"
    assert any(marker is not None for marker in active_by_region), "at least one subregion must be filled"

    states = _states_from_regions(active_by_region)
    assert all(states[i] != states[i + 1] for i in range(len(states) - 1)), "every ray must separate distinct states"

    assert distractors, "need outside distractors so containment matters"
    for r, c, color in distractors:
        assert not _inside_frame(frame_box, r, c), "distractors must be outside the frame interior"
        assert int(grid[r, c]) == color

    expected = _oracle_output(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver must match the structural oracle"

    ungated = expected.copy()
    for r, c, color in distractors:
        ungated[r, c] = color
    assert not np.array_equal(ungated, expected), "outside distractors must visibly matter"


def _gen_attempt() -> np.ndarray | None:
    n_rays = random.randint(2, 4)
    n_regions = n_rays + 1
    pattern = random.choice(_occupancy_patterns(n_regions))
    region_widths = _sample_region_widths(n_regions)
    frame_h = random.randint(*FRAME_HEIGHT_RANGE)
    frame_w = sum(region_widths) + n_rays + 2

    max_extra_h = min(CANVAS_EXTRA_RANGE[1], 30 - frame_h)
    max_extra_w = min(CANVAS_EXTRA_RANGE[1], 30 - frame_w)
    if max_extra_h < CANVAS_EXTRA_RANGE[0] or max_extra_w < CANVAS_EXTRA_RANGE[0]:
        return None

    canvas_h = frame_h + random.randint(CANVAS_EXTRA_RANGE[0], max_extra_h)
    canvas_w = frame_w + random.randint(CANVAS_EXTRA_RANGE[0], max_extra_w)

    top_margin = random.randint(1, canvas_h - frame_h - 1)
    left_margin = random.randint(1, canvas_w - frame_w - 1)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + frame_h - 1
    c1, split_cols, regions = _build_frame_columns(c0, region_widths)
    assert c1 == c0 + frame_w - 1
    frame_box = (r0, c0, r1, c1)

    split_colors = random.sample(PALETTE, n_rays)
    active_region_indices = [index for index, occupied in enumerate(pattern) if occupied]
    active_colors = random.sample([color for color in PALETTE if color not in split_colors], len(active_region_indices))

    active_by_region: list[Marker | None] = [None] * n_regions
    for color, region_index in zip(active_colors, active_region_indices):
        left, right = regions[region_index]
        marker = (
            random.randint(r0 + 1, r1 - 1),
            random.randint(left, right),
            color,
        )
        active_by_region[region_index] = marker

    strict, loose = _outside_candidates((canvas_h, canvas_w), frame_box)
    distractor_positions = _pick_spaced_points(strict[:], len(active_colors))
    if distractor_positions is None:
        distractor_positions = _pick_spaced_points(loose[:], len(active_colors))
    if distractor_positions is None:
        return None
    distractors = [
        (r, c, color)
        for (r, c), color in zip(distractor_positions, active_colors)
    ]

    grid = _draw_input_grid(
        (canvas_h, canvas_w),
        frame_box,
        split_cols,
        split_colors,
        active_by_region,
        distractors,
    )

    try:
        _validate_case(grid, active_by_region, distractors)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
