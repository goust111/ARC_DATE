import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse four stacked same-size color masks into one mask by overlaying them with a fixed band priority.

    Input structure
    - The grid is split into four equal-height horizontal bands.
    - Each band is a same-size mask on black background and has exactly one
      foreground color.

    Rule
    - At each local position, output the first foreground pixel found in priority
      order: first band, second band, fourth band, third band.
    - If all four corresponding positions are black, output black.

    Key insight
    - The bands are transparent layers sharing the same coordinate system; the
      output is their composite with the lower two bands resolved in bottom-up order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 4 == 0, "expected four equal-height horizontal bands"

    band_height = height // 4
    assert band_height > 0 and width > 0, "expected nonempty bands"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    bands = [
        g[i * band_height : (i + 1) * band_height, :]
        for i in range(4)
    ]
    band_colors = []
    for band in bands:
        colors = colors_present(band, exclude={bg})
        assert len(colors) == 1, "expected one foreground color per band"
        color = int(colors[0])
        assert np.all((band == bg) | (band == color)), "expected monochrome mask bands"
        band_colors.append(color)
    assert len(set(band_colors)) == 4, "expected distinct band colors"

    out = np.full((band_height, width), bg, dtype=int)
    for band_index in (2, 3, 1, 0):
        band = bands[band_index]
        mask = band != bg
        out[mask] = band[mask]

    return out

# --------------- generator (o5x) ---------------

# Geometry: four stacked 3x6 masks produce one 3x6 output.
_GEN_BAND_COUNT = 4
_GEN_BAND_H = 3
_GEN_WIDTH = 6

# Palette and overlay priority: colors are fixed to match the eval pattern.
_GEN_COLORS = [5, 4, 2, 8]
_GEN_PRIORITY = (2, 3, 1, 0)

# Per-band mask constraints.
_GEN_MIN_PIXELS = 4
_GEN_MAX_PIXELS = 14
_GEN_MAX_COMPONENTS = 5

# Output and retry constraints.
_GEN_MAX_OUTPUT_BLACK = 3
_GEN_ATTEMPTS = 300

_GEN_CELLS = [(r, c) for r in range(_GEN_BAND_H) for c in range(_GEN_WIDTH)]


def _gen_choice(items):
    return items[int(np.random.randint(0, len(items)))]


def _gen_shuffled(items):
    order = np.random.permutation(len(items))
    return [items[int(i)] for i in order]


def _gen_component_count(mask):
    return len(components_mask(mask, connectivity=4))


def _gen_random_mask(pixel_count):
    mask = np.zeros((_GEN_BAND_H, _GEN_WIDTH), dtype=bool)
    for r, c in _gen_shuffled(_GEN_CELLS)[:pixel_count]:
        mask[r, c] = True

    while _gen_component_count(mask) > _GEN_MAX_COMPONENTS:
        if int(mask.sum()) >= _GEN_MAX_PIXELS:
            return None
        current = _gen_component_count(mask)
        choices = []
        best_count = current
        for r, c in _GEN_CELLS:
            if mask[r, c]:
                continue
            trial = mask.copy()
            trial[r, c] = True
            new_count = _gen_component_count(trial)
            if new_count <= best_count:
                if new_count < best_count:
                    choices = []
                    best_count = new_count
                choices.append((r, c))
        if not choices:
            return None
        r, c = _gen_choice(choices)
        mask[r, c] = True
    return mask


def _gen_add_keeps_components(mask, r, c):
    if mask[r, c]:
        return True
    trial = mask.copy()
    trial[r, c] = True
    return _gen_component_count(trial) <= _GEN_MAX_COMPONENTS


def _gen_make_masks():
    masks = []
    for _ in range(_GEN_BAND_COUNT):
        pixel_count = int(np.random.randint(_GEN_MIN_PIXELS, _GEN_MAX_PIXELS + 1))
        mask = _gen_random_mask(pixel_count)
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _gen_fill_output_holes(masks):
    """Mutate masks until at most three overlay positions remain black."""
    black_cells = [(int(r), int(c)) for r, c in np.argwhere(~np.logical_or.reduce(masks))]
    for r, c in _gen_shuffled(black_cells):
        if int((~np.logical_or.reduce(masks)).sum()) <= _GEN_MAX_OUTPUT_BLACK:
            break
        choices = []
        for band_i, mask in enumerate(masks):
            if int(mask.sum()) >= _GEN_MAX_PIXELS:
                continue
            if _gen_add_keeps_components(mask, r, c):
                choices.append(band_i)
        if not choices:
            return False
        masks[_gen_choice(choices)][r, c] = True
    return True


def _gen_render_input(masks):
    inp = np.zeros((_GEN_BAND_H * 4, _GEN_WIDTH), dtype=int)
    for band_i, color in enumerate(_GEN_COLORS):
        band = inp[band_i * _GEN_BAND_H : (band_i + 1) * _GEN_BAND_H]
        band[masks[band_i]] = color
    return inp


def _gen_expected_output(masks):
    expected = np.zeros((_GEN_BAND_H, _GEN_WIDTH), dtype=int)
    for band_i in _GEN_PRIORITY:
        expected[masks[band_i]] = _GEN_COLORS[band_i]
    return expected


def _gen_validates(inp, expected):
    out = main(inp)
    return int((out == 0).sum()) <= _GEN_MAX_OUTPUT_BLACK and np.array_equal(out, expected)


def _gen_attempt():
    masks = _gen_make_masks()
    if masks is None:
        return None
    if not _gen_fill_output_holes(masks):
        return None

    inp = _gen_render_input(masks)
    expected = _gen_expected_output(masks)
    if not _gen_validates(inp, expected):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 3d31c5b3 input")
