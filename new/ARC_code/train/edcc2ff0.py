from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import components_mask, paste, random_free_location
import itertools
import random


TOP_BG = 0


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected a nonempty region"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _lower_panel_start(g: np.ndarray) -> int:
    full_non_black_rows = np.flatnonzero(np.all(g != TOP_BG, axis=1))
    assert full_non_black_rows.size, "expected a solid non-black lower panel"
    panel_start = int(full_non_black_rows[0])
    assert 0 < panel_start < g.shape[0], "expected a top instruction strip above the panel"
    assert np.all(g[panel_start:] != TOP_BG), "lower panel must contain no black cells"
    return panel_start


def _markers(top: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(top != TOP_BG)
    assert coords.size, "expected marker pixels in the instruction strip"
    assert np.all(coords[:, 1] == 0), "marker pixels must be in the first column"

    rows = [int(r) for r in coords[:, 0]]
    assert len(set(rows)) == len(rows), "expected at most one marker per row"

    markers = [(r, int(top[r, 0])) for r in sorted(rows)]
    colors = [color for _row, color in markers]
    assert len(set(colors)) == len(colors), "marker colors must be unique"
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the lower-panel colors named by the top markers and expand each marker into an object-count bar.

    Input structure
    A black instruction strip sits above a non-black rectangular scene panel.  The strip contains one-cell markers
    in the first column, and each marker color is a candidate object color in the lower panel.  The lower panel has
    a unique dominant background color and scattered 4-connected foreground objects.

    Rule
    In the lower panel, preserve objects whose colors appear as top markers and recolor every other foreground
    object to the panel background.  In the instruction strip, replace each marker by a horizontal run from column 0
    whose length is the number of 4-connected lower-panel objects of that marker color; erase markers with count 0.

    Key insight
    The top strip is a color filter plus a compact count summary: marker colors select which lower-panel objects
    survive, and their bars report the number of surviving objects by color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panel_start = _lower_panel_start(g)
    top = g[:panel_start]
    bottom = g[panel_start:]

    markers = _markers(top)
    marker_colors = {color for _row, color in markers}
    panel_bg = _unique_mode(bottom)
    assert panel_bg not in marker_colors, "marker colors must be foreground colors in the lower panel"

    out = np.zeros_like(g)
    cleaned = bottom.copy()
    for color in set(map(int, np.unique(bottom))) - marker_colors - {panel_bg}:
        cleaned[cleaned == color] = panel_bg
    out[panel_start:] = cleaned

    width = g.shape[1]
    for row, color in markers:
        count = len(components_mask(bottom == color, connectivity=4))
        assert count <= width, "object count must fit as a horizontal marker bar"
        out[row, :count] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and instruction-strip geometry.
_GEN_HEIGHT = 20
_GEN_WIDTH = 10
_GEN_TOP_H = 7
_GEN_PANEL_H = 13
_GEN_MARKER_ROWS = (1, 3, 5)

# Palette and object-count ranges.
_GEN_COLORS = tuple(range(1, 10))
_GEN_ACTIVE_MARKER_COUNTS = (2, 3)
_GEN_TOTAL_OBJECTS_RANGE = (6, 11)
_GEN_OBJECTS_PER_MARKER_RANGE = (1, 5)
_GEN_DISTRACTOR_COUNT_RANGE = (1, 3)
_GEN_DISTRACTOR_COLOR_RANGE = (1, 2)

# Blob area and shape sampling.
_GEN_BLOB_AREAS = (2, 3, 4)
_GEN_FG_AREA_RANGE = (20, 26)
_GEN_AREA_SHAPES = {
    2: ((1, 2), (2, 1)),
    3: ((1, 3), (3, 1)),
    4: ((1, 4), (4, 1), (2, 2)),
}

# Placement and retry limits.
_GEN_PLACEMENT_BORDER = 1
_GEN_PLACEMENT_PADDING = 1
_GEN_PLACEMENT_CONNECTIVITY = 4
_GEN_RETRY_LIMIT = 300


def _gen_count_split(total, parts):
    min_count, max_count = _GEN_OBJECTS_PER_MARKER_RANGE
    splits = itertools.product(range(min_count, max_count + 1), repeat=parts)
    choices = [vals for vals in splits if sum(vals) == total]
    return list(random.choice(choices))


def _gen_rect_specs(count):
    min_area = max(_GEN_FG_AREA_RANGE[0], _GEN_BLOB_AREAS[0] * count)
    max_area = min(_GEN_FG_AREA_RANGE[1], _GEN_BLOB_AREAS[-1] * count)
    target_area = random.randint(min_area, max_area)
    area_products = itertools.product(_GEN_BLOB_AREAS, repeat=count)
    area_choices = [vals for vals in area_products if sum(vals) == target_area]
    areas = list(random.choice(area_choices))
    specs = [random.choice(_GEN_AREA_SHAPES[area]) for area in areas]
    random.shuffle(specs)
    return specs


def _gen_color_counts():
    distractor_count = random.randint(*_GEN_DISTRACTOR_COUNT_RANGE)
    active_count = random.choice(_GEN_ACTIVE_MARKER_COUNTS)
    total_min, total_max = _GEN_TOTAL_OBJECTS_RANGE
    per_marker_max = _GEN_OBJECTS_PER_MARKER_RANGE[1]
    marker_sum_min = max(active_count, total_min - distractor_count)
    marker_sum_max = min(_GEN_WIDTH, total_max - distractor_count, per_marker_max * active_count)
    marker_sum = random.randint(marker_sum_min, marker_sum_max)

    active_slots = random.sample(range(len(_GEN_MARKER_ROWS)), active_count)
    active_counts = _gen_count_split(marker_sum, active_count)
    counts = [0] * len(_GEN_MARKER_ROWS)
    for slot, count in zip(active_slots, active_counts):
        counts[slot] = count
    return counts, distractor_count


def _gen_distractor_objects(panel_bg, marker_colors, total):
    min_colors, max_colors = _GEN_DISTRACTOR_COLOR_RANGE
    color_count = random.randint(min_colors, min(max_colors, total))
    colors = random.sample(
        [c for c in _GEN_COLORS if c not in {panel_bg, *marker_colors}],
        color_count,
    )
    objects = list(colors)
    for _ in range(total - color_count):
        objects.append(random.choice(colors))
    random.shuffle(objects)
    return objects


def _gen_place_blobs(panel, objects, specs):
    for color, shape in zip(objects, specs):
        patch = np.full(shape, int(color), dtype=int)
        try:
            r, c = random_free_location(
                panel,
                patch,
                transparent=int(panel[0, 0]),
                border=_GEN_PLACEMENT_BORDER,
                padding=_GEN_PLACEMENT_PADDING,
                padding_connectivity=_GEN_PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return False
        paste(panel, patch, r, c)
    return True


def _gen_output(panel_bg, marker_colors, counts, marker_objects, marker_specs):
    panel = np.full((_GEN_PANEL_H, _GEN_WIDTH), panel_bg, dtype=int)
    if not _gen_place_blobs(panel, marker_objects, marker_specs):
        return None

    out = np.zeros((_GEN_HEIGHT, _GEN_WIDTH), dtype=int)
    out[_GEN_TOP_H:] = panel
    for row, color, count in zip(_GEN_MARKER_ROWS, marker_colors, counts):
        out[row, :count] = color
    return out


def _gen_input(out, marker_colors, distractor_objects, distractor_specs):
    inp = out.copy()
    inp[:_GEN_TOP_H] = TOP_BG
    for row, color in zip(_GEN_MARKER_ROWS, marker_colors):
        inp[row, 0] = color

    inp_panel = inp[_GEN_TOP_H:]
    if not _gen_place_blobs(inp_panel, distractor_objects, distractor_specs):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    panel_bg = random.choice(_GEN_COLORS)
    marker_colors = random.sample([c for c in _GEN_COLORS if c != panel_bg], len(_GEN_MARKER_ROWS))
    counts, distractor_total = _gen_color_counts()
    marker_objects = [color for color, count in zip(marker_colors, counts) for _ in range(count)]
    random.shuffle(marker_objects)
    distractor_objects = _gen_distractor_objects(panel_bg, marker_colors, distractor_total)

    rect_specs = _gen_rect_specs(len(marker_objects) + len(distractor_objects))
    split_at = len(marker_objects)
    marker_specs = rect_specs[:split_at]
    distractor_specs = rect_specs[split_at:]

    out = _gen_output(panel_bg, marker_colors, counts, marker_objects, marker_specs)
    if out is None:
        return None
    inp = _gen_input(out, marker_colors, distractor_objects, distractor_specs)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid edcc2ff0 input")
