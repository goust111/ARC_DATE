from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, paste
import random


def _separator_line(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    height, width = map(int, g.shape)
    candidates: list[tuple[str, int, int]] = []
    for color in colors_present(g, exclude={bg}):
        for row in range(height):
            if np.all(g[row, :] == color):
                candidates.append(("h", row, int(color)))
        for col in range(width):
            if np.all(g[:, col] == color):
                candidates.append(("v", col, int(color)))

    assert len(candidates) == 1, "expected exactly one full foreground separator line"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the colored object across the full separator line while swapping its two colors in the original location.

    Input structure
    - A background grid contains exactly one full foreground row or column separator.
    - All other foreground pixels form a source mask entirely on one side of the separator.
    - The source mask uses exactly two colors, and its mirror positions on the other side are background.

    Rule
    - Copy each source pixel to its reflection across the full separator line, preserving its original color.
    - Recolor the source pixel in place by swapping the two source colors.
    - Leave the separator and all background cells otherwise unchanged.

    Key insight
    - The separator is an axis of reflection, and the asymmetric two-color source mask is both copied across the axis and color-inverted at its starting side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    height, width = map(int, g.shape)

    orient, axis, sep_color = _separator_line(g, bg)
    source_mask = (g != bg) & (g != sep_color)
    source_coords = [(int(r), int(c)) for r, c in np.argwhere(source_mask)]
    assert source_coords, "expected source pixels away from the separator"

    source_colors = set(map(int, g[source_mask]))
    assert len(source_colors) == 2, "source mask must use exactly two colors"
    color_a, color_b = tuple(source_colors)
    swapped = {color_a: color_b, color_b: color_a}

    out = g.copy()
    if orient == "h":
        above = sum(r < axis for r, _c in source_coords)
        below = sum(r > axis for r, _c in source_coords)
        assert above + below == len(source_coords), "source pixels cannot lie on the separator"
        assert (above == 0) ^ (below == 0), "source mask must be entirely on one side"

        for r, c in source_coords:
            mirror_r = 2 * axis - r
            assert 0 <= mirror_r < height, "reflected row must remain in bounds"
            assert int(g[mirror_r, c]) == bg, "reflected target cell must be background"
            out[mirror_r, c] = int(g[r, c])
            out[r, c] = swapped[int(g[r, c])]
    else:
        left = sum(c < axis for _r, c in source_coords)
        right = sum(c > axis for _r, c in source_coords)
        assert left + right == len(source_coords), "source pixels cannot lie on the separator"
        assert (left == 0) ^ (right == 0), "source mask must be entirely on one side"

        for r, c in source_coords:
            mirror_c = 2 * axis - c
            assert 0 <= mirror_c < width, "reflected column must remain in bounds"
            assert int(g[r, mirror_c]) == bg, "reflected target cell must be background"
            out[r, mirror_c] = int(g[r, c])
            out[r, c] = swapped[int(g[r, c])]

    return out

# --------------- generator (o5x) ---------------

# Canvas and separator layout.
BG_COLOR, SEP_COLOR = 0, 1
GRID_SIZE_MIN, GRID_SIZE_MAX = 9, 16
MIN_SIDE_DEPTH = 4
SEPARATOR_GAP = 1

# Source sprite shape and coloring.
SOURCE_COLORS = range(2, 10)
SOURCE_COLOR_COUNT = 2
SPRITE_PIXELS_MIN, SPRITE_PIXELS_MAX = 7, 11
SPRITE_AXIS_SIZE_MIN, SPRITE_AXIS_SIZE_MAX = 3, 5
SPRITE_FREE_SIZES = (3, 4, 5)
MIN_EMPTY_CELLS_IN_BBOX = 1
SPLIT_COMPONENT_PROB = 0.35
RANDOM_COLOR_PARTITION_PROB = 0.65

# Split-component sprite construction.
MIN_COMPONENT_PIXELS, SPLIT_GAP_CELLS = 2, 1
CONNECTED_COMPONENTS, SPLIT_COMPONENTS = 1, 2

# Generation retry limit.
MAX_GENERATION_ATTEMPTS = 300


def _gen_frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    frontier: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r + 1 < h and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c + 1 < w and mask[r, c + 1])
            ):
                frontier.append((r, c))
    return frontier


def _gen_grow_from_seed(h: int, w: int, count: int, seed: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[seed] = True
    while int(mask.sum()) < count:
        mask[random.choice(_gen_frontier_cells(mask))] = True
    return mask


def _gen_connected_mask(h: int, w: int, count: int) -> np.ndarray:
    corners = [
        ((0, 0), (h - 1, w - 1)),
        ((0, w - 1), (h - 1, 0)),
        ((h - 1, 0), (0, w - 1)),
        ((h - 1, w - 1), (0, 0)),
    ]
    (r, c), (end_r, end_c) = random.choice(corners)
    dr = 1 if end_r > r else -1
    dc = 1 if end_c > c else -1
    steps = [(dr, 0)] * abs(end_r - r) + [(0, dc)] * abs(end_c - c)
    random.shuffle(steps)

    mask = np.zeros((h, w), dtype=bool)
    mask[r, c] = True
    for sr, sc in steps:
        r += sr
        c += sc
        mask[r, c] = True

    while int(mask.sum()) < count:
        mask[random.choice(_gen_frontier_cells(mask))] = True
    return mask


def _gen_split_options(h: int, w: int, count: int) -> list[tuple[str, int, int, int]]:
    options: list[tuple[str, int, int, int]] = []
    for split in range(SPLIT_GAP_CELLS, w - SPLIT_GAP_CELLS):
        left_cap = h * split
        right_cap = h * (w - split - SPLIT_GAP_CELLS)
        for n1 in range(MIN_COMPONENT_PIXELS, count - MIN_COMPONENT_PIXELS + 1):
            n2 = count - n1
            if MIN_COMPONENT_PIXELS <= n2 <= right_cap and n1 <= left_cap:
                options.append(("v", split, n1, n2))

    for split in range(SPLIT_GAP_CELLS, h - SPLIT_GAP_CELLS):
        top_cap = split * w
        bottom_cap = (h - split - SPLIT_GAP_CELLS) * w
        for n1 in range(MIN_COMPONENT_PIXELS, count - MIN_COMPONENT_PIXELS + 1):
            n2 = count - n1
            if MIN_COMPONENT_PIXELS <= n2 <= bottom_cap and n1 <= top_cap:
                options.append(("h", split, n1, n2))
    return options


def _gen_split_mask(h: int, w: int, count: int) -> np.ndarray:
    orient, split, n1, n2 = random.choice(_gen_split_options(h, w, count))
    mask = np.zeros((h, w), dtype=bool)
    if orient == "v":
        left_w = split
        right_w = w - split - SPLIT_GAP_CELLS
        if random.choice([False, True]):
            seed1 = (0, 0)
            seed2 = (h - 1, right_w - 1)
        else:
            seed1 = (h - 1, 0)
            seed2 = (0, right_w - 1)
        mask[:, :left_w] = _gen_grow_from_seed(h, left_w, n1, seed1)
        mask[:, split + SPLIT_GAP_CELLS :] = _gen_grow_from_seed(h, right_w, n2, seed2)
    else:
        top_h = split
        bottom_h = h - split - SPLIT_GAP_CELLS
        if random.choice([False, True]):
            seed1 = (0, 0)
            seed2 = (bottom_h - 1, w - 1)
        else:
            seed1 = (0, w - 1)
            seed2 = (bottom_h - 1, 0)
        mask[:top_h, :] = _gen_grow_from_seed(top_h, w, n1, seed1)
        mask[split + SPLIT_GAP_CELLS :, :] = _gen_grow_from_seed(bottom_h, w, n2, seed2)
    return mask


def _gen_sprite_choices(axis_sizes, free_sizes, axis_is_rows, count):
    choices: list[tuple[int, int, int]] = []
    for axis_size in axis_sizes:
        for free_size in free_sizes:
            h, w = (axis_size, free_size) if axis_is_rows else (free_size, axis_size)
            if h * w < count + MIN_EMPTY_CELLS_IN_BBOX:
                continue
            if h + w - 1 <= count:
                choices.append((h, w, CONNECTED_COMPONENTS))
            if _gen_split_options(h, w, count):
                choices.append((h, w, SPLIT_COMPONENTS))
    return choices


def _gen_choose_first_color_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if random.random() < RANDOM_COLOR_PARTITION_PROB:
        cut = random.randint(1, len(coords) - 1)
        return coords[:cut]

    by_axis = sorted(coords, key=lambda rc: rc[random.randrange(2)])
    cut = random.randint(1, len(by_axis) - 1)
    return by_axis[:cut]


def _gen_sprite(axis_sizes, free_sizes, axis_is_rows, colors) -> np.ndarray | None:
    count = random.randint(SPRITE_PIXELS_MIN, SPRITE_PIXELS_MAX)
    want_split = random.random() < SPLIT_COMPONENT_PROB
    choices = _gen_sprite_choices(axis_sizes, free_sizes, axis_is_rows, count)
    if not choices:
        return None

    split_choices = [choice for choice in choices if choice[2] == SPLIT_COMPONENTS]
    connected_choices = [choice for choice in choices if choice[2] == CONNECTED_COMPONENTS]
    if want_split and split_choices:
        h, w, comp_count = random.choice(split_choices)
    elif connected_choices:
        h, w, comp_count = random.choice(connected_choices)
    else:
        h, w, comp_count = random.choice(split_choices)

    mask = _gen_connected_mask(h, w, count) if comp_count == CONNECTED_COMPONENTS else _gen_split_mask(h, w, count)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    random.shuffle(coords)
    first = _gen_choose_first_color_coords(coords)

    sprite = np.zeros((h, w), dtype=int)
    sprite[mask] = colors[1]
    for r, c in first:
        sprite[r, c] = colors[0]
    return sprite


def _gen_swapped_sprite(sprite: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    out = sprite.copy()
    out[sprite == a] = b
    out[sprite == b] = a
    return out


def _gen_canvas_and_separator() -> tuple[int, int, str, int, tuple[int, int]]:
    height = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    orient = random.choice(["h", "v"])
    if orient == "h":
        axis = random.randint(MIN_SIDE_DEPTH, height - MIN_SIDE_DEPTH - 1)
        side_depths = (axis, height - axis - 1)
    else:
        axis = random.randint(MIN_SIDE_DEPTH, width - MIN_SIDE_DEPTH - 1)
        side_depths = (axis, width - axis - 1)
    return height, width, orient, axis, side_depths


def _gen_source_location(height, width, orient, axis, side_depths, sprite) -> tuple[int, int] | None:
    sh, sw = map(int, sprite.shape)
    source_on_low_side = random.choice([False, True])
    if orient == "h":
        if source_on_low_side:
            r_min = max(0, axis - side_depths[1])
            r_max = axis - sh - SEPARATOR_GAP
        else:
            r_min = axis + 1 + SEPARATOR_GAP
            r_max = min(height - sh, axis + side_depths[0] - sh)
        c_min = 0
        c_max = width - sw
    else:
        r_min = 0
        r_max = height - sh
        if source_on_low_side:
            c_min = max(0, axis - side_depths[1])
            c_max = axis - sw - SEPARATOR_GAP
        else:
            c_min = axis + 1 + SEPARATOR_GAP
            c_max = min(width - sw, axis + side_depths[0] - sw)

    if r_min > r_max or c_min > c_max:
        return None
    return random.randint(r_min, r_max), random.randint(c_min, c_max)


def _gen_render_output(height, width, orient, axis, sprite, colors, r0, c0):
    sh, sw = map(int, sprite.shape)
    out = np.zeros((height, width), dtype=int)
    if orient == "h":
        out[axis, :] = SEP_COLOR
        reflected = np.flipud(sprite)
        rr0 = 2 * axis - (r0 + sh - 1)
        rc0 = c0
    else:
        out[:, axis] = SEP_COLOR
        reflected = np.fliplr(sprite)
        rr0 = r0
        rc0 = 2 * axis - (c0 + sw - 1)

    paste(out, _gen_swapped_sprite(sprite, colors), r0, c0, transparent=BG_COLOR)
    paste(out, reflected, rr0, rc0, transparent=BG_COLOR)
    return out, reflected, rr0, rc0


def _gen_strip_reflection(out, sprite, reflected, r0, c0, rr0, rc0):
    inp = out.copy()
    ref_mask = reflected != 0
    inp[rr0 : rr0 + reflected.shape[0], rc0 : rc0 + reflected.shape[1]][ref_mask] = BG_COLOR
    paste(inp, sprite, r0, c0, transparent=BG_COLOR)
    return inp


def _gen_validated_input(inp: np.ndarray, out: np.ndarray) -> np.ndarray | None:
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def _gen_attempt() -> np.ndarray | None:
    height, width, orient, axis, side_depths = _gen_canvas_and_separator()
    axis_max = min(SPRITE_AXIS_SIZE_MAX, min(side_depths) - SEPARATOR_GAP)
    axis_sizes = list(range(SPRITE_AXIS_SIZE_MIN, axis_max + 1))
    colors = tuple(random.sample(SOURCE_COLORS, SOURCE_COLOR_COUNT))
    sprite = _gen_sprite(axis_sizes, list(SPRITE_FREE_SIZES), orient == "h", colors)
    if sprite is None:
        return None

    location = _gen_source_location(height, width, orient, axis, side_depths, sprite)
    if location is None:
        return None
    r0, c0 = location

    out, reflected, rr0, rc0 = _gen_render_output(height, width, orient, axis, sprite, colors, r0, c0)
    inp = _gen_strip_reflection(out, sprite, reflected, r0, c0, rr0, rc0)
    return _gen_validated_input(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
