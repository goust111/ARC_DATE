import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect the multi-cell foreground mask across each singleton marker so the marker becomes one cell of its colored copy.

    Input structure
    A background canvas contains one multi-pixel foreground object and one or more singleton marker colors. Each marker
    shares exactly one row or exactly one column with a pixel of the source object.

    Rule
    Keep the source object. For each marker, if it shares a row with exactly one source pixel, reflect the whole source
    object horizontally through the vertical midline between that source pixel and the marker. If it shares a column
    with exactly one source pixel, reflect the whole source object vertically through the horizontal midline between
    them. Paint the reflected mask in the marker color.

    Key insight
    The singleton marker is one already-reflected cell of the source object, so its aligned source pixel fixes the
    mirror axis for the entire copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    coords_by_color = {color: np.argwhere(g == color) for color in colors}

    source_colors = [color for color, coords in coords_by_color.items() if len(coords) > 1]
    assert len(source_colors) == 1
    source_color = source_colors[0]
    source_coords = coords_by_color[source_color]

    marker_colors = [color for color in colors if color != source_color]
    assert marker_colors

    out = g.copy()
    height, width = g.shape
    for marker_color in marker_colors:
        marker_coords = coords_by_color[marker_color]
        assert len(marker_coords) == 1
        marker_r, marker_c = map(int, marker_coords[0])

        row_anchors = source_coords[source_coords[:, 0] == marker_r]
        col_anchors = source_coords[source_coords[:, 1] == marker_c]
        use_horizontal_reflection = len(row_anchors) == 1
        use_vertical_reflection = len(col_anchors) == 1
        assert use_horizontal_reflection != use_vertical_reflection

        targets = []
        if use_horizontal_reflection:
            anchor_c = int(row_anchors[0, 1])
            col_sum = marker_c + anchor_c
            for r, c in source_coords:
                targets.append((int(r), int(col_sum - c)))
        else:
            anchor_r = int(col_anchors[0, 0])
            row_sum = marker_r + anchor_r
            for r, c in source_coords:
                targets.append((int(row_sum - r), int(c)))

        for r, c in targets:
            assert 0 <= r < height and 0 <= c < width
            assert out[r, c] in (bg, marker_color)
        for r, c in targets:
            out[r, c] = marker_color

    return out

# --------------- generator (o5x) ---------------

# Canvas dimensions.
GRID_HEIGHT_RANGE = (11, 14)
GRID_WIDTH_RANGE = (11, 16)

# Source shape filters.
SOURCE_COLOR = 1
SOURCE_PIXEL_COUNTS = tuple(range(6, 10))
SOURCE_BBOX_HEIGHTS = tuple(range(3, 5))
SOURCE_BBOX_WIDTHS = tuple(range(3, 6))
SOURCE_DENSITY_RANGE = (0.44, 0.58)

# Marker and reflected-copy layout.
BACKGROUND_COLOR = 0
MARKER_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
MARKER_COUNTS = (1, 2)
MARKER_GAPS = tuple(range(4, 10))
REFLECTION_SIDES = (-1, 1)

# Generation retry budget.
GENERATION_RETRIES = 300

_GEN_SOURCE_CACHE = {}


def _gen_source_cache():
    if "by_count" in _GEN_SOURCE_CACHE:
        return _GEN_SOURCE_CACHE["by_count"]

    by_count = {n: [] for n in SOURCE_PIXEL_COUNTS}
    for bh in SOURCE_BBOX_HEIGHTS:
        for bw in SOURCE_BBOX_WIDTHS:
            area = bh * bw
            for n in SOURCE_PIXEL_COUNTS:
                density = n / area
                if not (SOURCE_DENSITY_RANGE[0] <= density <= SOURCE_DENSITY_RANGE[1]):
                    continue
                for combo in itertools.combinations(range(area), n):
                    mask = np.zeros((bh, bw), dtype=bool)
                    mask.ravel()[list(combo)] = True
                    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
                        continue
                    if not is_contiguous(mask.astype(int), background=0, connectivity=8):
                        continue
                    if not ((mask.sum(axis=1) == 1).any() or (mask.sum(axis=0) == 1).any()):
                        continue
                    by_count[n].append(mask)

    _GEN_SOURCE_CACHE["by_count"] = by_count
    return by_count


def _gen_source_shape():
    by_count = _gen_source_cache()
    counts = [n for n, masks in by_count.items() if masks]
    n = random.choice(counts)
    return random.choice(by_count[n]).copy()


def _axis_counts(source_set):
    row_counts = {}
    col_counts = {}
    for r, c in source_set:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1
    return row_counts, col_counts


def _axis_marker_options(axis, source_set, ar, ac):
    """Build all marker/reflection options for one anchor and reflection axis."""
    options = []
    for side in REFLECTION_SIDES:
        for gap in MARKER_GAPS:
            d = gap + 1
            if axis == "H":
                marker = (ar, ac + side * d)
                target = tuple((r, 2 * ac + side * d - c) for r, c in source_set)
            else:
                marker = (ar + side * d, ac)
                target = tuple((2 * ar + side * d - r, c) for r, c in source_set)

            target_set = set(target)
            if not target_set.isdisjoint(source_set):
                continue
            options.append(
                {
                    "axis": axis,
                    "anchor": (ar, ac),
                    "gap": gap,
                    "side": side,
                    "marker": marker,
                    "target": target,
                    "target_set": target_set,
                }
            )
    return options


def _gen_marker_options(src_coords):
    source_set = {tuple(map(int, xy)) for xy in src_coords}
    row_counts, col_counts = _axis_counts(source_set)
    options = []
    for ar, ac in source_set:
        if row_counts[ar] == 1:
            options.extend(_axis_marker_options("H", source_set, ar, ac))
        if col_counts[ac] == 1:
            options.extend(_axis_marker_options("V", source_set, ar, ac))
    return options


def _gen_bounds(source_set, marker_set, height, width):
    occupied = set(source_set)
    for opt in marker_set:
        occupied.update(opt["target_set"])
    rs = [r for r, _ in occupied]
    cs = [c for _, c in occupied]
    r_min, r_max = min(rs), max(rs)
    c_min, c_max = min(cs), max(cs)
    if r_max - r_min + 1 > height or c_max - c_min + 1 > width:
        return None
    return -r_min, height - 1 - r_max, -c_min, width - 1 - c_max


def _gen_marker_sets(source_set, options, height, width, count):
    feasible = []
    if count == 1:
        for opt in options:
            bounds = _gen_bounds(source_set, (opt,), height, width)
            if bounds is not None:
                feasible.append(((opt,), bounds))
        return feasible

    for opt_a, opt_b in itertools.combinations(options, 2):
        if opt_a["marker"] == opt_b["marker"] or not opt_a["target_set"].isdisjoint(opt_b["target_set"]):
            continue
        marker_set = (opt_a, opt_b)
        bounds = _gen_bounds(source_set, marker_set, height, width)
        if bounds is not None:
            feasible.append((marker_set, bounds))
    return feasible


def _choose_marker_design(source_set, options, height, width):
    """Choose a feasible marker set and placement bounds for the source-local design."""
    feasible_by_count = {}
    for count in MARKER_COUNTS:
        feasible = _gen_marker_sets(source_set, options, height, width, count)
        if feasible:
            feasible_by_count[count] = feasible
    if not feasible_by_count:
        return None

    marker_count = random.choice(list(feasible_by_count))
    return random.choice(feasible_by_count[marker_count])


def _render_reflection_pair(height, width, source_set, marker_set, bounds):
    """Render both oracle output and stripped input for one placed design."""
    r0 = random.randint(bounds[0], bounds[1])
    c0 = random.randint(bounds[2], bounds[3])
    marker_colors = random.sample(MARKER_COLORS, len(marker_set))

    out = np.zeros((height, width), dtype=int)
    for r, c in source_set:
        out[r0 + r, c0 + c] = SOURCE_COLOR

    for opt, color in zip(marker_set, marker_colors):
        for r, c in opt["target"]:
            out[r0 + r, c0 + c] = color
        mr, mc = opt["marker"]
        out[r0 + mr, c0 + mc] = color

    inp = out.copy()
    for opt in marker_set:
        marker = opt["marker"]
        for r, c in opt["target"]:
            if (r, c) != marker:
                inp[r0 + r, c0 + c] = BACKGROUND_COLOR

    return inp, out


def _gen_attempt():
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    source = _gen_source_shape()
    src_coords = [tuple(map(int, xy)) for xy in np.argwhere(source)]
    source_set = set(src_coords)
    options = _gen_marker_options(src_coords)
    if not options:
        return None

    marker_design = _choose_marker_design(source_set, options, height, width)
    if marker_design is None:
        return None
    marker_set, bounds = marker_design

    inp, out = _render_reflection_pair(height, width, source_set, marker_set, bounds)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 963f59bc input")
