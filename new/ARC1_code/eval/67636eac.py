import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each foreground sprite to its bounding box and pack the crops tightly along the sprites' layout axis.

    Input structure:
    - The input is a background grid containing multiple separated, monochrome, 8-connected sprites.
    - The sprites all have the same bounding-box shape.
    - The sprite centers are spread more in either rows or columns, defining a vertical or horizontal layout.

    Rule:
    - Extract each sprite's bbox crop.
    - If the sprite centers span more columns than rows, order sprites left-to-right and concatenate the crops horizontally.
    - Otherwise, order sprites top-to-bottom and concatenate the crops vertically.

    Key insight:
    - The output is not a recoloring; it is the same sprites with all empty space between their bboxes removed.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    sprites = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    assert len(sprites) >= 2

    items = []
    shapes = set()
    centers = []
    for sprite in sprites:
        bbox = tuple(map(int, sprite["bbox"]))
        crop = crop_bbox(g, bbox)
        fg = crop[crop != bg]
        assert fg.size == int(sprite["area"])
        assert len(set(map(int, fg))) == 1

        r0, c0, r1, c1 = bbox
        items.append((bbox, crop))
        shapes.add(tuple(crop.shape))
        centers.append(((r0 + r1) / 2, (c0 + c1) / 2))

    assert len(shapes) == 1
    sprite_h, sprite_w = next(iter(shapes))

    row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
    col_span = max(c for _, c in centers) - min(c for _, c in centers)
    assert row_span != col_span

    if col_span > row_span:
        starts = [bbox[1] for bbox, _ in items]
        assert len(set(starts)) == len(starts)
        ordered = sorted(items, key=lambda item: item[0][1])

        out = np.full((sprite_h, sprite_w * len(ordered)), bg, dtype=int)
        for i, (_, crop) in enumerate(ordered):
            paste(out, crop, 0, i * sprite_w)
    else:
        starts = [bbox[0] for bbox, _ in items]
        assert len(set(starts)) == len(starts)
        ordered = sorted(items, key=lambda item: item[0][0])

        out = np.full((sprite_h * len(ordered), sprite_w), bg, dtype=int)
        for i, (_, crop) in enumerate(ordered):
            paste(out, crop, i * sprite_h, 0)

    return out

# --------------- generator (o5x) ---------------

_GEN_RNG = np.random.default_rng()

# Canvas bounds and margins.
GRID_ROW_RANGE = (12, 19)
GRID_COL_RANGE = (10, 19)
EDGE_MARGIN = 1
# Sprite count, geometry, and colors.
SPRITE_COUNT_RANGE = (2, 5)
SPRITE_SIZE = 3
COLOR_IDS = np.arange(1, 10)
# LR+UD symmetric 3x3 pattern orbits and sampled orbit combinations.
PATTERN_GROUPS = (
    ((0, 0), (0, 2), (2, 0), (2, 2)),
    ((0, 1), (2, 1)), ((1, 0), (1, 2)), ((1, 1),),
)
PATTERN_GROUP_CHOICES = ((1, 2), (0, 3), (1, 2, 3), (0, 1, 3), (0, 2, 3))
# Layout spacing constraints.
PAIR_AXIS_MIN_LENGTH = 10
PAIR_AXIS_STEP = 5
MULTI_AXIS_STEP = 4
MAX_CROSS_SPAN = 4
# Generation retry budget.
GEN_RETRY_LIMIT = 300

def _gen_pattern(rng):
    pat = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    choice = PATTERN_GROUP_CHOICES[int(rng.integers(0, len(PATTERN_GROUP_CHOICES)))]
    for group_i in choice:
        for r, c in PATTERN_GROUPS[group_i]:
            pat[r, c] = 1
    return pat

def _max_start(length): return int(length - SPRITE_SIZE - EDGE_MARGIN)

def _min_axis_length(n):
    multi = EDGE_MARGIN + MULTI_AXIS_STEP * (n - 1) + SPRITE_SIZE + EDGE_MARGIN
    return PAIR_AXIS_MIN_LENGTH if n == 2 else multi

def _gen_axis_length(rng, n, horizontal):
    canvas_low = GRID_COL_RANGE[0] if horizontal else GRID_ROW_RANGE[0]
    canvas_high = GRID_COL_RANGE[1] if horizontal else GRID_ROW_RANGE[1]
    low = max(canvas_low, _min_axis_length(n))
    if low >= canvas_high:
        return None
    return int(rng.integers(low, canvas_high))

def _gen_axis_starts(rng, n, length):
    if n == 2:
        base_steps = np.array([PAIR_AXIS_STEP], dtype=int)
    else:
        base_steps = np.full(n - 1, MULTI_AXIS_STEP, dtype=int)
    slack = int(length - (2 * EDGE_MARGIN + SPRITE_SIZE) - base_steps.sum())
    if slack < 0:
        return None

    lead = 0
    extra_steps = np.zeros(n - 1, dtype=int)
    if slack > 0:
        if n >= 3:
            forced_i = int(rng.integers(0, n - 1))
            forced = int(rng.integers(1, slack + 1))
            extra_steps[forced_i] += forced
            slack -= forced
        for _ in range(slack):
            bucket = int(rng.integers(0, n + 1))
            if bucket == 0:
                lead += 1
            elif bucket < n:
                extra_steps[bucket - 1] += 1

    starts = [EDGE_MARGIN + lead]
    for step in base_steps + extra_steps:
        starts.append(starts[-1] + int(step))
    if starts[-1] > _max_start(length):
        return None
    return starts

def _gen_cross_starts(rng, n, length):
    lo, hi = EDGE_MARGIN, _max_start(length)
    span_max = min(MAX_CROSS_SPAN, hi - lo)
    if span_max <= 0:
        return [lo] * n
    span = int(rng.integers(0, span_max + 1))
    if n >= 3:
        span = max(1, span)
    base = int(rng.integers(lo, hi - span + 1))
    vals = [base + int(rng.integers(0, span + 1)) for _ in range(n)]
    if n >= 3 and span > 0 and len(set(vals)) == 1:
        vals[-1] = base if vals[0] != base else base + span
    return vals

def _gen_grid_shape(rng, n, horizontal):
    if horizontal:
        h = int(rng.integers(*GRID_ROW_RANGE))
        w = _gen_axis_length(rng, n, True)
    else:
        h = _gen_axis_length(rng, n, False)
        w = int(rng.integers(*GRID_COL_RANGE))
    return h, w

def _build_output(pattern, colors, horizontal):
    sprites = [pattern * color for color in colors]
    out_shape = (
        (SPRITE_SIZE, SPRITE_SIZE * len(sprites))
        if horizontal
        else (SPRITE_SIZE * len(sprites), SPRITE_SIZE)
    )
    out = np.zeros(out_shape, dtype=int)
    for i, sprite in enumerate(sprites):
        if horizontal:
            paste(out, sprite, 0, SPRITE_SIZE * i)
        else:
            paste(out, sprite, SPRITE_SIZE * i, 0)
    return sprites, out

def _scatter_sprites(shape, sprites, axis_starts, cross_starts, horizontal):
    inp = np.zeros(shape, dtype=int)
    for sprite, a0, x0 in zip(sprites, axis_starts, cross_starts):
        if horizontal:
            paste(inp, sprite, x0, a0, transparent=0)
        else:
            paste(inp, sprite, a0, x0, transparent=0)
    return inp

def _gen_attempt() -> np.ndarray | None:
    rng = _GEN_RNG
    n = int(rng.integers(*SPRITE_COUNT_RANGE))
    horizontal = bool(rng.integers(0, 2))
    colors = [int(c) for c in rng.choice(COLOR_IDS, size=n, replace=False)]
    pattern = _gen_pattern(rng)

    h, w = _gen_grid_shape(rng, n, horizontal)
    if h is None or w is None:
        return None

    sprites, out = _build_output(pattern, colors, horizontal)

    axis_len = w if horizontal else h
    cross_len = h if horizontal else w
    axis_starts = _gen_axis_starts(rng, n, axis_len)
    if axis_starts is None:
        return None
    cross_starts = _gen_cross_starts(rng, n, cross_len)

    inp = _scatter_sprites((h, w), sprites, axis_starts, cross_starts, horizontal)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None

def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 67636eac")
