import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize each colored frame as a row whose length is the number of gray markers it contains.

    Input structure
    The input has a dominant background, several 4-connected colored frame
    objects, and gray marker pixels placed inside those frames. Gray is not a
    frame color, and every marker lies inside exactly one colored frame bbox.

    Rule
    For each colored frame, count the gray marker pixels inside its bounding
    box. Sort the frames by this count from smallest to largest, then create
    one output row per frame: the frame color repeated for its marker count,
    padded on the right with background to the largest count.

    Key insight
    The output is not a spatial crop of the scene; it is a count summary where
    the gray marker population inside each frame determines both row length and
    row order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker = Color.GREY
    assert bg != marker, "gray markers must be foreground"
    assert np.any(g == marker), "expected gray marker pixels"

    frame_colors = colors_present(g, exclude={bg, marker})
    assert frame_colors, "expected at least one colored frame"
    frames = find_components(
        g,
        bg=bg,
        colors=frame_colors,
        connectivity=4,
        monochromatic=True,
    )
    assert frames, "expected colored frame components"

    marker_cover = np.zeros(g.shape, dtype=int)
    summaries: list[tuple[int, int]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        marker_mask = region == marker
        marker_count = int(marker_mask.sum())
        assert marker_count > 0, "each frame must contain at least one marker"

        marker_cover[r0 : r1 + 1, c0 : c1 + 1] += marker_mask.astype(int)
        summaries.append((marker_count, int(frame["color"])))

    assert np.all(marker_cover[g == marker] == 1), "every marker must belong to exactly one frame"
    counts = [count for count, _color in summaries]
    assert len(counts) == len(set(counts)), "marker counts must uniquely order the rows"

    summaries.sort(key=lambda item: item[0])
    out = np.full((len(summaries), max(counts)), bg, dtype=int)
    for row, (count, color) in enumerate(summaries):
        out[row, :count] = color
    return out

# --------------- generator (o5x) ---------------

_GEN_BG = 0
_GEN_MARKER = 5
_GEN_FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Canvas, counts, and colors.
_GEN_CANVAS_SIZE_RANGE = (20, 25)
_GEN_FRAME_COUNT_RANGE = (2, 4)
_GEN_MARKER_COUNT_RANGE = (2, 8)

# Frame geometry.
_GEN_BBOX_HEIGHT_RANGE = (5, 12)
_GEN_BBOX_WIDTH_RANGE = (5, 19)
_GEN_BORDER_RANGE = (1, 4)
_GEN_INTERIOR_HEIGHT_RANGE = (3, 10)
_GEN_INTERIOR_WIDTH_RANGE = (3, 15)
_GEN_MIN_EMPTY_INTERIOR_CELLS = 2

# Retry/cache.
_GEN_MAX_ATTEMPTS = 300
_GEN_SPEC_CACHE = {}


def _gen_span(bounds):
    return range(bounds[0], bounds[1] + 1)


def _gen_expected_output(items):
    ordered = sorted(items, key=lambda item: item[0])
    out = np.full((len(ordered), max(count for count, _ in ordered)), _GEN_BG, dtype=int)
    for row, (count, color) in enumerate(ordered):
        out[row, :count] = color
    return out


def _gen_feasible_frame_specs(marker_count):
    """Return bbox/border/interior specs that can hold this marker count."""
    if marker_count in _GEN_SPEC_CACHE:
        return _GEN_SPEC_CACHE[marker_count]

    min_ih, max_ih = _GEN_INTERIOR_HEIGHT_RANGE
    min_iw, max_iw = _GEN_INTERIOR_WIDTH_RANGE
    specs = []
    for bh in _gen_span(_GEN_BBOX_HEIGHT_RANGE):
        for bw in _gen_span(_GEN_BBOX_WIDTH_RANGE):
            for top in _gen_span(_GEN_BORDER_RANGE):
                for bottom in _gen_span(_GEN_BORDER_RANGE):
                    ih = bh - top - bottom
                    if not min_ih <= ih <= max_ih:
                        continue
                    for left in _gen_span(_GEN_BORDER_RANGE):
                        for right in _gen_span(_GEN_BORDER_RANGE):
                            iw = bw - left - right
                            has_room = ih * iw >= marker_count + _GEN_MIN_EMPTY_INTERIOR_CELLS
                            if min_iw <= iw <= max_iw and has_room:
                                specs.append((bh, bw, top, bottom, left, right, ih, iw))
    _GEN_SPEC_CACHE[marker_count] = specs
    return specs


def _gen_pick_frame_spec(marker_count):
    specs = _gen_feasible_frame_specs(marker_count)
    dims = sorted({(spec[0], spec[1]) for spec in specs})
    bh, bw = random.choice(dims)
    choices = [spec for spec in specs if spec[0] == bh and spec[1] == bw]
    return random.choice(choices)


def _gen_frame_patch(color, marker_count):
    bh, bw, top, bottom, left, right, ih, iw = _gen_pick_frame_spec(marker_count)
    patch = np.full((bh, bw), _GEN_BG, dtype=int)
    patch[:top, :] = color
    patch[bh - bottom :, :] = color
    patch[:, :left] = color
    patch[:, bw - right :] = color
    for cell in random.sample(range(ih * iw), marker_count):
        rr, cc = divmod(cell, iw)
        patch[top + rr, left + cc] = _GEN_MARKER
    return patch


def _gen_scene_plan():
    """Sample canvas size, frame colors, marker counts, and expected output."""
    size = random.randint(*_GEN_CANVAS_SIZE_RANGE)
    n_frames = random.randint(*_GEN_FRAME_COUNT_RANGE)
    colors = random.sample(_GEN_FRAME_COLORS, n_frames)
    counts = random.sample(_gen_span(_GEN_MARKER_COUNT_RANGE), n_frames)
    items = list(zip(counts, colors))
    return size, items, _gen_expected_output(items)


def _gen_place_patches(size, patches):
    """Place non-overlapping frame patches, or return None if placement fails."""
    grid = np.full((size, size), _GEN_BG, dtype=int)
    occupied = np.full_like(grid, _GEN_BG)
    order = sorted(range(len(patches)), key=lambda i: patches[i].size, reverse=True)

    for i in order:
        patch = patches[i]
        box = np.ones(patch.shape, dtype=int)
        try:
            r, c = random_free_location(occupied, box, transparent=_GEN_BG)
        except ValueError:
            return None
        paste(occupied, box, r, c)
        paste(grid, patch, r, c)

    return grid


def _gen_validated_grid(grid, expected):
    """Run the solver oracle and reject generated grids that do not round-trip."""
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def _gen_attempt() -> np.ndarray | None:
    size, items, expected = _gen_scene_plan()
    patches = [_gen_frame_patch(color, count) for count, color in items]
    grid = _gen_place_patches(size, patches)
    if grid is None:
        return None
    return _gen_validated_grid(grid, expected)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d4c90558 input")
