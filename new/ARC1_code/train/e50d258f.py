import numpy as np

from arc_utils import Color, crop_bbox, find_components, paste as _gen_paste, random_free_location as _gen_random_free_location
import itertools as _gen_itertools
import random as _gen_random


def _solid_panel_boxes(g: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates = []
    for color in np.unique(g):
        bg = int(color)
        components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
        if len(components) < 2:
            continue

        boxes = []
        valid = True
        for component in components:
            box = tuple(map(int, component["bbox"]))
            crop = crop_bbox(g, box)
            if int(component["area"]) != crop.size or np.any(crop == bg):
                valid = False
                break
            boxes.append(box)

        if valid:
            candidates.append((bg, boxes))

    assert len(candidates) == 1, "expected one background color that separates solid panels"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the solid rectangular panel that contains the most red marker pixels.

    Input structure
    - The grid has one separator/background color whose complement forms multiple
      4-connected solid rectangular panels.
    - Each panel is a complete rectangle made from the same visible palette, with
      red and blue marker pixels embedded among teal cells.

    Rule
    - Count red marker pixels inside each panel.
    - Return the exact bounding-box crop of the unique panel with the largest red
      marker count.

    Key insight
    - The output is already present in the input; the only operation is selecting
      the panel by its red-marker count and cropping it unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, boxes = _solid_panel_boxes(g)
    assert len(boxes) >= 2

    scored = []
    for box in boxes:
        panel = crop_bbox(g, box)
        assert np.all(panel != bg)
        red_count = int(np.count_nonzero(panel == Color.RED))
        assert red_count > 0, "expected red markers in each panel"
        scored.append((red_count, box))

    best_count = max(red_count for red_count, _box in scored)
    winners = [box for red_count, box in scored if red_count == best_count]
    assert len(winners) == 1, "expected a unique panel with the most red markers"
    return crop_bbox(g, winners[0])

# --------------- generator (o5x) ---------------

# Palette.
_GEN_BLACK = 0
_GEN_BLUE = 1
_GEN_RED = 2
_GEN_TEAL = 8

# Canvas and panel geometry.
_GEN_CANVAS_SHAPE = (10, 10)
_GEN_PANEL_COUNTS = (2, 3)
_GEN_PANEL_HEIGHTS, _GEN_PANEL_WIDTHS = range(3, 8), range(3, 7)
_GEN_COVERAGE_CELLS = (46, 64)
_GEN_DIMS = [(h, w) for h in _GEN_PANEL_HEIGHTS for w in _GEN_PANEL_WIDTHS]

# Marker counts and placement rejection limits.
_GEN_MIN_RED, _GEN_MIN_WINNER_RED, _GEN_MAX_WINNER_RED = 1, 2, 6
_GEN_RED_WIN_MARGIN, _GEN_MIN_BLUE, _GEN_MAX_BLUE = 1, 1, 5
_GEN_TEAL_MARGIN, _GEN_PLACEMENT_PADDING, _GEN_PADDING_CONNECTIVITY = 1, 1, 8
_GEN_MAX_ATTEMPTS = 300


def _gen_choose_dims(panel_count):
    min_cells, max_cells = _GEN_COVERAGE_CELLS
    choices = [
        combo
        for combo in _gen_itertools.combinations(_GEN_DIMS, int(panel_count))
        if min_cells <= sum(h * w for h, w in combo) <= max_cells
    ]
    dims = list(_gen_random.choice(choices))
    _gen_random.shuffle(dims)
    return dims


def _gen_red_counts(areas):
    caps = [(int(area) - _GEN_MIN_BLUE - _GEN_TEAL_MARGIN) // 2 for area in areas]
    winner = _gen_random.randrange(len(areas))
    runner_cap = max(caps[i] for i in range(len(areas)) if i != winner)
    max_red = _gen_random.randint(
        _GEN_MIN_WINNER_RED, min(_GEN_MAX_WINNER_RED, caps[winner], runner_cap + _GEN_RED_WIN_MARGIN)
    )
    runner_min = max_red - _GEN_RED_WIN_MARGIN
    runner = _gen_random.choice([i for i in range(len(areas)) if i != winner and caps[i] >= runner_min])

    counts = []
    for i, cap in enumerate(caps):
        if i == winner:
            counts.append(max_red)
        elif i == runner:
            counts.append(runner_min)
        else:
            counts.append(_gen_random.randint(_GEN_MIN_RED, min(cap, runner_min)))
    return counts, winner


def _gen_blue_count(area, red_count):
    area = int(area)
    red_count = int(red_count)
    teal_over_red = area - 2 * red_count - _GEN_TEAL_MARGIN
    teal_over_blue = (area - red_count - _GEN_TEAL_MARGIN) // 2
    return _gen_random.randint(_GEN_MIN_BLUE, min(_GEN_MAX_BLUE, teal_over_red, teal_over_blue))


def _gen_make_panel(h, w, red_count):
    h, w, red_count = int(h), int(w), int(red_count)
    area = h * w
    blue_count = _gen_blue_count(area, red_count)
    panel = np.full((h, w), _GEN_TEAL, dtype=int)
    cells = np.array(_gen_random.sample(range(area), red_count + blue_count), dtype=int)
    red_cells = cells[:red_count]
    blue_cells = cells[red_count:]
    panel[red_cells // w, red_cells % w] = _GEN_RED
    panel[blue_cells // w, blue_cells % w] = _GEN_BLUE
    return panel


def _gen_place_panels(panels):
    canvas = np.full(_GEN_CANVAS_SHAPE, _GEN_BLACK, dtype=int)
    order = sorted(range(len(panels)), key=lambda i: panels[i].size, reverse=True)
    for i in order:
        try:
            r, c = _gen_random_free_location(
                canvas, panels[i], transparent=_GEN_BLACK, padding=_GEN_PLACEMENT_PADDING, padding_connectivity=_GEN_PADDING_CONNECTIVITY
            )
        except ValueError:
            return None
        _gen_paste(canvas, panels[i], r, c)
    return canvas


def _gen_attempt():
    panel_count = _gen_random.choice(_GEN_PANEL_COUNTS)
    dims = _gen_choose_dims(panel_count)
    red_counts, winner = _gen_red_counts([h * w for h, w in dims])
    panels = [_gen_make_panel(h, w, red_counts[i]) for i, (h, w) in enumerate(dims)]
    canvas = _gen_place_panels(panels)
    if canvas is None:
        return None
    try:
        if not np.array_equal(main(canvas), panels[winner]):
            return None
    except Exception:
        return None
    return canvas


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e50d258f input")
