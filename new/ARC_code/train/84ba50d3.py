from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg
import random


def _divider_row(g: np.ndarray, bg: int) -> tuple[int, int]:
    rows: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        row_colors = set(map(int, g[r]))
        if len(row_colors) == 1:
            color = next(iter(row_colors))
            if int(color) != int(bg):
                rows.append((int(r), int(color)))
    assert len(rows) == 1, "expected exactly one solid non-background divider row"
    return rows[0]


def _unique_widest_row(coords: np.ndarray) -> int:
    row_counts: list[tuple[int, int]] = []
    for r in sorted(set(map(int, coords[:, 0]))):
        row_cols = np.sort(coords[coords[:, 0] == r, 1])
        assert np.array_equal(row_cols, np.arange(int(row_cols[0]), int(row_cols[-1]) + 1))
        row_counts.append((int(row_cols.size), int(r)))

    max_count = max(count for count, _r in row_counts)
    widest = [r for count, r in row_counts if count == max_count]
    assert max_count > 1, "multi-column sprites must have a horizontal span"
    assert len(widest) == 1, "expected a unique widest horizontal row"
    return int(widest[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the blue sprites downward relative to the red divider, with vertical sprites passing through to the bottom.

    Input structure:
    - The background is the dominant color.
    - One non-background color forms a single full-width horizontal divider row.
    - The other non-background color forms 4-connected sprites above the divider.

    Rule:
    - Clear the original sprite pixels.
    - A one-column sprite falls to the bottom of its column and deletes the divider
      pixel it crossed.
    - Any wider sprite falls until its unique widest horizontal row is immediately
      above the divider row; if the moved sprite overlaps the divider, it covers it.

    Key insight:
    The red row is not a generic floor: only one-column sprites pass through to the
    bottom, while wider sprites are registered by their widest horizontal segment.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    line_r, line_color = _divider_row(g, bg)

    non_bg_colors = set(map(int, g.ravel())) - {bg}
    sprite_colors = non_bg_colors - {line_color}
    assert len(sprite_colors) == 1, "expected exactly one sprite color"
    sprite_color = next(iter(sprite_colors))

    sprites = find_components(
        g,
        bg=bg,
        colors=[sprite_color],
        connectivity=4,
        monochromatic=True,
    )
    assert sprites, "expected at least one sprite"

    out = g.copy()
    out[g == sprite_color] = bg
    occupied = np.zeros((H, W), dtype=bool)

    for sprite in sprites:
        coords = sprite["coords"]
        _r0, c0, r1, c1 = map(int, sprite["bbox"])
        assert r1 < line_r, "all sprites must begin above the divider"

        if c0 == c1:
            shift = H - 1 - r1
            assert shift >= 0
            assert int(out[line_r, c0]) == line_color
            out[line_r, c0] = bg
        else:
            widest_r = _unique_widest_row(coords)
            shift = line_r - 1 - widest_r
            assert shift >= 0

        moved = coords.copy()
        moved[:, 0] += shift
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < H))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < W))

        for rr, cc in moved:
            r = int(rr)
            c = int(cc)
            assert not occupied[r, c], "moved sprites must not overlap"
            occupied[r, c] = True
            out[r, c] = sprite_color

    return out

# --------------- generator (att2) ---------------

# Canvas, palette, and divider bounds.
MIN_GRID_SIZE, LARGE_GRID_MIN, MAX_GRID_SIZE = 7, 14, 18
LARGE_GRID_PROB, PALETTE_SIZE, DIVIDER_MARGIN = 0.35, 10, 3
ADJACENT_ROW_OFFSET = 1

# Sprite counts, spacing, and retry limits.
MIN_SPRITE_COUNT, MAX_SPRITE_COUNT = 1, 5
MAX_SINGLE_SPRITES, MAX_MULTI_SPRITES = 2, 3
GAP_COLUMNS, INPUT_GAP_TO_DIVIDER = 1, 1
GENERATION_RETRIES = 300

# Sprite geometry.
MIN_SPRITE_HEIGHT, MAX_SINGLE_HEIGHT, MAX_MULTI_HEIGHT = 1, 7, 5
SINGLE_WIDTH, MIN_MULTI_WIDTH, MAX_MULTI_WIDTH = 1, 2, 3
WIDE_MULTI_PROB = 0.5


def _composition(total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[random.randrange(parts)] += 1
    return shares


def _min_layout_width(n_single: int, n_multi: int) -> int:
    n = n_single + n_multi
    return n_single * SINGLE_WIDTH + MIN_MULTI_WIDTH * n_multi + GAP_COLUMNS * max(0, n - 1)


def _count_pairs(width: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for n_single in range(MAX_SINGLE_SPRITES + 1):
        for n_multi in range(MAX_MULTI_SPRITES + 1):
            n = n_single + n_multi
            if MIN_SPRITE_COUNT <= n <= MAX_SPRITE_COUNT and _min_layout_width(n_single, n_multi) <= width:
                pairs.append((n_single, n_multi))
    return pairs


def _place_intervals(widths: list[int], grid_width: int) -> list[int]:
    n = len(widths)
    base = sum(widths) + GAP_COLUMNS * max(0, n - 1)
    extra = grid_width - base
    assert extra >= 0
    gaps = _composition(extra, n + 1)

    starts: list[int] = []
    c = gaps[0]
    for i, width in enumerate(widths):
        starts.append(c)
        c += width
        if i < n - 1:
            c += GAP_COLUMNS + gaps[i + 1]
    return starts


def _multi_coords(top: int, left: int, height: int, width: int, bar_row: int, spine_col: int) -> np.ndarray:
    coords = {(top + r, left + spine_col) for r in range(height)}
    coords.update((top + bar_row, left + c) for c in range(width))
    return np.array(sorted(coords), dtype=int)


def _build_descriptors(n_single: int, n_multi: int, max_single_h: int, max_multi_h: int, grid_width: int) -> list[dict[str, int | str]]:
    descriptors: list[dict[str, int | str]] = []
    for _ in range(n_single):
        descriptors.append(
            {"kind": "single", "width": SINGLE_WIDTH, "height": random.randint(MIN_SPRITE_HEIGHT, max_single_h)}
        )

    extra_width = grid_width - _min_layout_width(n_single, n_multi)
    for _ in range(n_multi):
        width = MIN_MULTI_WIDTH
        if extra_width > 0 and random.random() < WIDE_MULTI_PROB:
            width = MAX_MULTI_WIDTH
            extra_width -= MAX_MULTI_WIDTH - MIN_MULTI_WIDTH
        height = random.randint(MIN_SPRITE_HEIGHT, max_multi_h)
        descriptors.append(
            {
                "kind": "multi",
                "width": width,
                "height": height,
                "bar_row": random.randint(0, height - ADJACENT_ROW_OFFSET),
                "spine_col": random.randrange(width),
            }
        )

    random.shuffle(descriptors)
    return descriptors


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(LARGE_GRID_MIN, MAX_GRID_SIZE) if random.random() < LARGE_GRID_PROB else random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    W = H
    bg, divider_color, sprite_color = random.sample(range(PALETTE_SIZE), 3)
    divider_choices = list(range(DIVIDER_MARGIN, H - DIVIDER_MARGIN))
    divider_weights = [min(r - ADJACENT_ROW_OFFSET, H - r - ADJACENT_ROW_OFFSET) for r in divider_choices]
    divider_r = random.choices(divider_choices, weights=divider_weights, k=1)[0]

    pairs = _count_pairs(W)
    if not pairs:
        return None
    n_single, n_multi = random.choice(pairs)

    max_single_h = min(MAX_SINGLE_HEIGHT, divider_r - INPUT_GAP_TO_DIVIDER, H - divider_r - ADJACENT_ROW_OFFSET)
    max_multi_h = min(MAX_MULTI_HEIGHT, divider_r - INPUT_GAP_TO_DIVIDER)
    if (n_single and max_single_h < MIN_SPRITE_HEIGHT) or (n_multi and max_multi_h < MIN_SPRITE_HEIGHT):
        return None

    descriptors = _build_descriptors(n_single, n_multi, max_single_h, max_multi_h, W)
    starts = _place_intervals([int(d["width"]) for d in descriptors], W)

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)
    inp[divider_r, :] = divider_color
    out[divider_r, :] = divider_color

    input_max_bottom = divider_r - INPUT_GAP_TO_DIVIDER - ADJACENT_ROW_OFFSET
    for desc, left in zip(descriptors, starts):
        height = int(desc["height"])
        width = int(desc["width"])
        if desc["kind"] == "single":
            col = left
            out[divider_r, col] = bg
            out[H - height : H, col] = sprite_color

            input_bottom = random.randint(height - ADJACENT_ROW_OFFSET, input_max_bottom)
            inp[input_bottom - height + ADJACENT_ROW_OFFSET : input_bottom + ADJACENT_ROW_OFFSET, col] = sprite_color
            continue

        bar_row = int(desc["bar_row"])
        spine_col = int(desc["spine_col"])
        landed_top = divider_r - ADJACENT_ROW_OFFSET - bar_row
        landed = _multi_coords(landed_top, left, height, width, bar_row, spine_col)
        if np.any((landed[:, 0] < 0) | (landed[:, 0] >= H)):
            return None
        out[landed[:, 0], landed[:, 1]] = sprite_color

        landed_bottom = int(landed[:, 0].max())
        min_shift = landed_bottom - input_max_bottom
        max_shift = landed_top
        if min_shift > max_shift:
            return None
        shift = random.randint(min_shift, max_shift)
        original = landed.copy()
        original[:, 0] -= shift
        if np.any((original[:, 0] < 0) | (original[:, 0] > input_max_bottom)):
            return None
        inp[original[:, 0], original[:, 1]] = sprite_color

    bg_count = int(np.count_nonzero(inp == bg))
    if bg_count <= int(np.count_nonzero(inp == divider_color)):
        return None
    if bg_count <= int(np.count_nonzero(inp == sprite_color)):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 84ba50d3 input")
