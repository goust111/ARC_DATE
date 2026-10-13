import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, is_contiguous as _gen_is_contiguous, paste, paste as _gen_paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the four scattered 3x3 foreground sprites into a compact 2x2 atlas.

    Input structure
    - A dominant background color and exactly one foreground color.
    - Four separated 8-connected foreground objects, each contained in a 3x3 bbox.

    Rule
    - Crop each object's 3x3 bbox, preserving its internal background holes.
    - Order the crops by their input positions: two upper sprites, then two lower sprites;
      within each pair, left before right.
    - Paste them into a 7x7 background canvas as 3x3 blocks with one blank row and one
      blank column between the blocks.

    Key insight
    - The input is an unpacked 2x2 sprite atlas; 8-connectivity keeps diagonal-only
      sprite pixels in the same object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1

    components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) == 4

    sprites = []
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3)
        crop = crop_bbox(g, (r0, c0, r1, c1))
        assert set(map(int, np.unique(crop))).issubset({bg, int(fg_colors[0])})
        sprites.append({"bbox": (r0, c0, r1, c1), "crop": crop})

    by_row = sorted(sprites, key=lambda s: (s["bbox"][0], s["bbox"][1]))
    assert by_row[1]["bbox"][0] < by_row[2]["bbox"][0]

    ordered_rows = []
    for row in (by_row[:2], by_row[2:]):
        ordered = sorted(row, key=lambda s: s["bbox"][1])
        assert ordered[0]["bbox"][3] < ordered[1]["bbox"][1]
        ordered_rows.append(ordered)

    out = np.full((7, 7), bg, dtype=int)
    for out_r, row in zip((0, 4), ordered_rows):
        for out_c, sprite in zip((0, 4), row):
            paste(out, sprite["crop"], out_r, out_c)

    return out

# --------------- generator (o5x) ---------------
# Sprite, canvas, and retry distribution knobs.
_GEN_BG, _GEN_FG = 0, 2
_GEN_SPRITE_SIZE, _GEN_SPRITE_COUNT, _GEN_PAIR_COUNT = 3, 4, 2
_GEN_MIN_FG_PIXELS, _GEN_MAX_FG_PIXELS = 4, 9
_GEN_INPUT_MIN_SIZE, _GEN_INPUT_MAX_SIZE = 16, 23
_GEN_OUTPUT_SIZE, _GEN_MAX_ATTEMPTS = 7, 300
_GEN_SPRITE_CELLS = _GEN_SPRITE_SIZE * _GEN_SPRITE_SIZE

# Input scatter gap and compact output atlas offsets.
_GEN_INPUT_EMPTY_GAP = 3
_GEN_LAYOUT_SPAN = 2 * _GEN_SPRITE_SIZE + _GEN_INPUT_EMPTY_GAP
_GEN_PAIR_OFFSET = _GEN_SPRITE_SIZE + _GEN_INPUT_EMPTY_GAP
_GEN_ATLAS_OFFSET = _GEN_SPRITE_SIZE + 1
_GEN_ATLAS_POSITIONS = ((0, 0), (0, _GEN_ATLAS_OFFSET), (_GEN_ATLAS_OFFSET, 0), (_GEN_ATLAS_OFFSET, _GEN_ATLAS_OFFSET))

_gen_rng, _gen_sprite_bank_cache = np.random.default_rng(), None


def _gen_randint(low, high):
    return int(_gen_rng.integers(low, high + 1))


def _gen_sprite_bank():
    global _gen_sprite_bank_cache
    if _gen_sprite_bank_cache is not None:
        return _gen_sprite_bank_cache

    bank = {n: [] for n in range(_GEN_MIN_FG_PIXELS, _GEN_MAX_FG_PIXELS + 1)}
    for bits in range(1, 1 << _GEN_SPRITE_CELLS):
        mask = np.array([(bits >> i) & 1 for i in range(_GEN_SPRITE_CELLS)], dtype=int).reshape((_GEN_SPRITE_SIZE,) * 2)
        n_fg = int(mask.sum())
        if not (_GEN_MIN_FG_PIXELS <= n_fg <= _GEN_MAX_FG_PIXELS):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _gen_is_contiguous(mask, background=_GEN_BG, connectivity=8):
            continue
        bank[n_fg].append(np.where(mask == 1, _GEN_FG, _GEN_BG).astype(int))

    _gen_sprite_bank_cache = bank
    return _gen_sprite_bank_cache


def _gen_sprite():
    n_fg = _gen_randint(_GEN_MIN_FG_PIXELS, _GEN_MAX_FG_PIXELS)
    choices = _gen_sprite_bank()[n_fg]
    return choices[_gen_randint(0, len(choices) - 1)].copy()


def _gen_render(shape, sprites, placements):
    """Paste sprites into a background canvas at top-left placements."""
    grid = np.full(shape, _GEN_BG, dtype=int)
    for sprite, (r, c) in zip(sprites, placements):
        _gen_paste(grid, sprite, r, c)
    return grid


def _gen_pair_cols(width):
    left = _gen_randint(0, width - _GEN_LAYOUT_SPAN)
    right = _gen_randint(left + _GEN_PAIR_OFFSET, width - _GEN_SPRITE_SIZE)
    return left, right


def _gen_input_placements(height, width):
    """Sample upper/lower row bands and separated column pairs."""
    upper_rows = [_gen_randint(0, height - _GEN_LAYOUT_SPAN) for _ in range(_GEN_PAIR_COUNT)]
    lower_min = max(upper_rows) + _GEN_PAIR_OFFSET
    lower_rows = [_gen_randint(lower_min, height - _GEN_SPRITE_SIZE) for _ in range(_GEN_PAIR_COUNT)]
    upper_cols = _gen_pair_cols(width)
    lower_cols = _gen_pair_cols(width)
    rows = (upper_rows[0], upper_rows[1], lower_rows[0], lower_rows[1])
    cols = (*upper_cols, *lower_cols)
    return tuple(zip(rows, cols))


def _gen_matches_solver(inp, expected):
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt():
    sprites = [_gen_sprite() for _ in range(_GEN_SPRITE_COUNT)]
    expected = _gen_render((_GEN_OUTPUT_SIZE, _GEN_OUTPUT_SIZE), sprites, _GEN_ATLAS_POSITIONS)
    height = _gen_randint(_GEN_INPUT_MIN_SIZE, _GEN_INPUT_MAX_SIZE)
    width = _gen_randint(_GEN_INPUT_MIN_SIZE, _GEN_INPUT_MAX_SIZE)
    inp = _gen_render((height, width), sprites, _gen_input_placements(height, width))
    return inp if _gen_matches_solver(inp, expected) else None


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 1990f7a8 input")
