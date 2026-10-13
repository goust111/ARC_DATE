from __future__ import annotations

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, is_contiguous as _gen_is_contiguous, paste, paste as _gen_paste
import itertools
import random


CONTROL = 5


def _filled_sprites(g: np.ndarray, bg: int) -> list[dict]:
    sprites = []
    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=False):
        r0, c0, r1, c1 = comp["bbox"]
        raw = crop_bbox(g, (r0, c0, r1, c1))
        colors = set(map(int, np.unique(raw))) - {int(bg), CONTROL}
        assert len(colors) == 1, "each marked sprite must contain exactly one non-grey color"
        color = colors.pop()
        assert np.any(raw == CONTROL), "each sprite must include grey control pixels"

        filled = raw.copy()
        filled[filled == CONTROL] = color
        sprites.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "raw": raw, "filled": filled})

    sprites.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for left, right in zip(sprites, sprites[1:]):
        assert left["bbox"][3] < right["bbox"][1], "sprite bboxes must be left-to-right separated"
    return sprites


def _top_offsets(sprites: list[dict]) -> list[int]:
    assert len(sprites) >= 2, "expected a chain of marked sprites"
    tops = [sprites[0]["bbox"][0]]
    for left, right in zip(sprites, sprites[1:]):
        left_rows = np.flatnonzero(left["raw"][:, -1] == CONTROL)
        right_rows = np.flatnonzero(right["raw"][:, 0] == CONTROL)
        assert len(left_rows) == 1, "left sprite needs one grey marker on its right edge"
        assert len(right_rows) == 1, "right sprite needs one grey marker on its left edge"
        tops.append(tops[-1] + int(left_rows[0]) - int(right_rows[0]))
    return tops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey-marked sprite to its own color and pack the sprites side by side with matching grey seam markers aligned.

    Input structure
    - The grid has three rows with black background.
    - The foreground is a left-to-right chain of separated 8-connected sprites.
    - Each sprite contains exactly one non-grey color plus grey control pixels.
    - Every seam has one grey control pixel on the previous sprite's right edge and one on the next sprite's left edge.

    Rule
    - Crop each sprite to its bounding box and replace all grey control pixels in that crop with the sprite's non-grey color.
    - Concatenate the cropped sprites from left to right, removing the background gap columns between their original bounding boxes.
    - Keep the first sprite at its original top row, then vertically place each next sprite so the two grey seam markers align on the same output row.

    Key insight
    - The grey pixels are both fill cells and connector instructions; aligning the grey edge markers determines the vertical offsets while the bbox widths determine the compact output width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    assert h == 3, "expected a three-row sprite chain"

    bg = get_bg(g)
    assert bg != CONTROL, "grey is the control color, not the background"
    sprites = _filled_sprites(g, bg)
    tops = _top_offsets(sprites)

    out_w = sum(int(item["filled"].shape[1]) for item in sprites)
    out = np.full((h, out_w), int(bg), dtype=int)
    c = 0
    for item, top in zip(sprites, tops):
        sprite = item["filled"]
        assert 0 <= top and top + sprite.shape[0] <= h, "aligned sprite must fit output height"
        paste(out, sprite, int(top), c, transparent=bg)
        c += int(sprite.shape[1])
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and sprite size bounds.
_GEN_BG = 0
_GEN_CONTROL = 5
_GEN_GRID_HEIGHT = 3
_GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_SPRITE_COUNTS = (3, 4)
_GEN_WIDTH_RANGE = range(2, 5)
_GEN_HEIGHT_RANGE = range(1, 4)
_GEN_OUTPUT_W_RANGE = range(7, 10)
_GEN_INPUT_W_RANGE = range(9, 12)
# Shape density, seam placement, and retry limits.
_GEN_MAX_HEIGHT_AREA = 7
_GEN_MAX_EMPTY_CELLS = 3
_GEN_MIN_COLORED_CELLS = 1
_GEN_MAX_COLORED_CELLS = 4
_GEN_CONNECTIVITY = 8
_GEN_INPUT_GAP = 1
_GEN_MAX_ATTEMPTS = 300


def _gen_widths(n: int) -> list[int]:
    choices = []
    for widths in itertools.product(_GEN_WIDTH_RANGE, repeat=n):
        total = sum(widths)
        input_w = total + _GEN_INPUT_GAP * (n - 1)
        if total in _GEN_OUTPUT_W_RANGE and input_w in _GEN_INPUT_W_RANGE:
            choices.append(widths)
    return list(random.choice(choices))


def _gen_height(width: int) -> int:
    valid = [h for h in _GEN_HEIGHT_RANGE if h * width <= _GEN_MAX_HEIGHT_AREA]
    return random.choice(valid)


def _gen_alignment(heights: list[int]) -> tuple[list[int], list[int]] | None:
    tops = [random.randint(0, _GEN_GRID_HEIGHT - heights[0])]
    seam_rows = []
    for h in heights[1:]:
        prev_top = tops[-1]
        prev_rows = set(range(prev_top, prev_top + heights[len(tops) - 1]))
        choices = []
        for top in range(0, _GEN_GRID_HEIGHT + 1 - h):
            overlap = sorted(prev_rows & set(range(top, top + h)))
            if overlap:
                choices.append((top, overlap))
        if not choices:
            return None
        top, overlap = random.choice(choices)
        tops.append(top)
        seam_rows.append(random.choice(overlap))
    return tops, seam_rows


def _gen_shape_cells(
    height: int, width: int, left_marker_row: int | None, right_marker_row: int | None
) -> set[tuple[int, int]] | None:
    required = set()
    if left_marker_row is not None:
        required.add((left_marker_row, 0))
    if right_marker_row is not None:
        required.add((right_marker_row, width - 1))

    area = height * width
    min_count = max(len(required), area - _GEN_MAX_EMPTY_CELLS, _GEN_MIN_COLORED_CELLS)
    max_count = min(_GEN_MAX_COLORED_CELLS, area)
    positions = [(r, c) for r in range(height) for c in range(width)]
    by_count: dict[int, list[set[tuple[int, int]]]] = {}

    for count in range(min_count, max_count + 1):
        for combo in itertools.combinations(positions, count):
            cells = set(combo)
            if not required <= cells:
                continue
            rows = [r for r, _c in cells]
            cols = [c for _r, c in cells]
            if min(rows) != 0 or max(rows) != height - 1 or min(cols) != 0 or max(cols) != width - 1:
                continue
            mask = np.zeros((height, width), dtype=int)
            mask[tuple(zip(*cells))] = 1
            if not _gen_is_contiguous(mask, background=_GEN_BG, connectivity=_GEN_CONNECTIVITY):
                continue
            by_count.setdefault(count, []).append(cells)

    if not by_count:
        return None
    count = random.choice(sorted(by_count))
    return set(random.choice(by_count[count]))


def _gen_layout() -> tuple[int, list[int], list[int], list[int], list[int | None], list[int | None]] | None:
    """Sample sprite dimensions and seam marker rows that can align in the output."""
    n_sprites = random.choice(_GEN_SPRITE_COUNTS)
    widths = _gen_widths(n_sprites)
    heights = [_gen_height(width) for width in widths]
    alignment = _gen_alignment(heights)
    if alignment is None:
        return None
    tops, seam_global_rows = alignment
    left_marker_rows: list[int | None] = [None] * n_sprites
    right_marker_rows: list[int | None] = [None] * n_sprites
    for i, seam_row in enumerate(seam_global_rows):
        right_marker_rows[i] = seam_row - tops[i]
        left_marker_rows[i + 1] = seam_row - tops[i + 1]
    return n_sprites, widths, heights, tops, left_marker_rows, right_marker_rows


def _gen_render_sprites(
    widths: list[int], heights: list[int], tops: list[int], colors: list[int], left_marker_rows: list[int | None],
    right_marker_rows: list[int | None],
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Build the packed output and marked input sprites from sampled layout choices."""
    out = np.full((_GEN_GRID_HEIGHT, sum(widths)), _GEN_BG, dtype=int)
    raw_sprites = []
    out_c = 0
    for i, (height, width, color) in enumerate(zip(heights, widths, colors)):
        cells = _gen_shape_cells(height, width, left_marker_rows[i], right_marker_rows[i])
        if cells is None:
            return None
        sprite = np.full((height, width), _GEN_BG, dtype=int)
        sprite[tuple(zip(*cells))] = color
        _gen_paste(out, sprite, tops[i], out_c, transparent=_GEN_BG)
        raw = sprite.copy()
        if left_marker_rows[i] is not None:
            raw[left_marker_rows[i], 0] = _GEN_CONTROL
        if right_marker_rows[i] is not None:
            raw[right_marker_rows[i], width - 1] = _GEN_CONTROL
        raw_sprites.append(raw)
        out_c += width
    return out, raw_sprites


def _gen_scatter_input(widths: list[int], tops: list[int], raw_sprites: list[np.ndarray]) -> np.ndarray:
    """Place marked sprites left-to-right with the fixed input gap."""
    inp_w = sum(widths) + _GEN_INPUT_GAP * (len(raw_sprites) - 1)
    inp = np.full((_GEN_GRID_HEIGHT, inp_w), _GEN_BG, dtype=int)
    inp_c = 0
    for i, raw in enumerate(raw_sprites):
        top = tops[0] if i == 0 else random.randint(0, _GEN_GRID_HEIGHT - raw.shape[0])
        _gen_paste(inp, raw, top, inp_c, transparent=_GEN_BG)
        inp_c += raw.shape[1] + _GEN_INPUT_GAP
    return inp


def _gen_attempt() -> np.ndarray | None:
    layout = _gen_layout()
    if layout is None:
        return None
    n_sprites, widths, heights, tops, left_marker_rows, right_marker_rows = layout
    colors = [random.choice(_GEN_COLORS) for _ in range(n_sprites)]
    rendered = _gen_render_sprites(widths, heights, tops, colors, left_marker_rows, right_marker_rows)
    if rendered is None:
        return None
    out, raw_sprites = rendered
    inp = _gen_scatter_input(widths, tops, raw_sprites)
    try:
        return inp if np.array_equal(main(inp), out) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 234bbc79 input")
