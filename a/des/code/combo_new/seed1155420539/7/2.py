from __future__ import annotations

import random

import numpy as np


BG = 0
ARC_MAX = 30
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return components


def _component_sprite(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    color = int(component["color"])
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in component["coords"]:
        sprite[int(r) - r0, int(c) - c0] = color
    return sprite


def _row_interval(mask: np.ndarray, row: int) -> tuple[int, int]:
    cols = np.flatnonzero(mask[row])
    assert cols.size > 0, "each fragment row must contain foreground"
    start = int(cols[0])
    end = int(cols[-1])
    assert end - start + 1 == int(cols.size), "fragment rows must be contiguous intervals"
    return start, end


def _assembled_rectangle_width(left_sprite: np.ndarray, right_sprite: np.ndarray) -> int:
    left_mask = left_sprite != BG
    right_mask = right_sprite != BG
    assert left_mask.shape[0] == right_mask.shape[0], "paired fragments must share a height"

    height = int(left_mask.shape[0])
    left_intervals = [_row_interval(left_mask, row) for row in range(height)]
    right_intervals = [_row_interval(right_mask, row) for row in range(height)]

    shift = left_intervals[0][1] + 1 - right_intervals[0][0]
    assert shift >= 0, "right fragment must stay on or to the right of the left fragment"

    starts = set()
    ends = set()
    for (left_start, left_end), (right_start, right_end) in zip(left_intervals, right_intervals):
        assert shift == left_end + 1 - right_start, "paired fragments must share one complementary seam"
        starts.add(int(left_start))
        ends.add(int(shift + right_end))

    assert len(starts) == 1 and len(ends) == 1, "assembled rows must span one common rectangle"
    return next(iter(ends)) - next(iter(starts)) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, interlock each pair into a rectangle, then pack the rectangles from shortest to tallest.

    Input structure:
    - The background is black.
    - The grid contains an even number of monochrome 4-connected foreground objects.
    - Each non-black color appears in exactly two objects, and the two same-colored objects have the same bounding-box height.
    - Within a same-colored pair, the two objects are complementary left/right fragments whose top-aligned union forms one solid rectangle.

    Rule:
    - Group the objects by color into pairs.
    - For each pair, use the left-right order from the input positions, align the two cropped fragments by their top rows, and horizontally interlock them so their complementary edges form a filled rectangle.
    - Replace each pair by that solid rectangle in the same color.
    - Sort the reconstructed rectangles by increasing height, then place them left to right with one black column between neighbors and all bottom edges aligned.

    Key insight:
    - The crucial step is recovering each pair's hidden rectangle from row-wise complementary boundaries; the final layout then depends only on the reconstructed rectangle heights, not on the original object positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    by_color: dict[int, list[dict]] = {}
    for component in _find_components(g):
        by_color.setdefault(int(component["color"]), []).append(component)

    assert by_color, "expected at least one color pair"

    rectangles: list[tuple[int, int, int, int]] = []
    for color, comps in by_color.items():
        assert len(comps) == 2, "each non-black color must appear exactly twice"
        comps = sorted(comps, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
        left_comp, right_comp = comps

        left_sprite = _component_sprite(left_comp)
        right_sprite = _component_sprite(right_comp)
        height = int(left_sprite.shape[0])
        assert height == int(right_sprite.shape[0]), "same-colored fragments must share one height"
        width = _assembled_rectangle_width(left_sprite, right_sprite)
        rectangles.append((height, int(left_comp["bbox"][1]), width, int(color)))

    rectangles.sort(key=lambda item: (item[0], item[1]))
    out_h = max(height for height, _left_x, _width, _color in rectangles)
    out_w = sum(width for _height, _left_x, width, _color in rectangles) + len(rectangles) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for height, _left_x, width, color in rectangles:
        out[out_h - height :, col : col + width] = int(color)
        col += width + 1

    return out


COLOR_POOL = tuple(range(1, 10))
PAIR_COUNT_CHOICES = (2, 3, 4)
HEIGHT_VALUES = tuple(range(2, 9))
RECT_WIDTH_RANGE = (4, 9)
CUT_STEP_CHOICES = (-2, -1, 0, 1, 2)

WORK_SHAPE = (ARC_MAX, ARC_MAX)
PLACEMENT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 500
PAIR_GAP_MIN = 2
EXTRA_MARGIN_MAX = 4


def _sample_cut_profile(height: int, width: int) -> list[int] | None:
    for _ in range(100):
        cut = random.randint(2, width - 2)
        cuts = [cut]
        for _row in range(1, height):
            cut = max(1, min(width - 1, cut + random.choice(CUT_STEP_CHOICES)))
            cuts.append(cut)

        if len(set(cuts)) < 2:
            continue
        if max(cuts) < 2 or min(cuts) > width - 2:
            continue
        return cuts if random.random() < 0.5 else list(reversed(cuts))
    return None


def _make_pair_sprites(height: int, width: int, color: int) -> tuple[np.ndarray, np.ndarray] | None:
    cuts = _sample_cut_profile(height, width)
    if cuts is None:
        return None

    left_w = max(cuts)
    right_w = width - min(cuts)
    left = np.full((height, left_w), BG, dtype=int)
    right = np.full((height, right_w), BG, dtype=int)
    min_cut = min(cuts)

    for row, cut in enumerate(cuts):
        left[row, :cut] = int(color)
        right[row, cut - min_cut :] = int(color)

    return left, right


def _sample_pair_specs() -> list[dict] | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    heights = random.sample(HEIGHT_VALUES, pair_count)
    widths = [random.randint(*RECT_WIDTH_RANGE) for _ in range(pair_count)]
    if sum(widths) + pair_count - 1 > ARC_MAX:
        return None

    colors = random.sample(COLOR_POOL, pair_count)
    specs = []
    for height, width, color in zip(heights, widths, colors):
        sprites = _make_pair_sprites(height, width, color)
        if sprites is None:
            return None
        left, right = sprites
        specs.append(
            {
                "height": int(height),
                "rect_width": int(width),
                "color": int(color),
                "left": left,
                "right": right,
            }
        )
    random.shuffle(specs)
    return specs


def _rectangles_separated(
    rects: list[tuple[int, int, int, int]],
    top: int,
    left: int,
    height: int,
    width: int,
) -> bool:
    r0 = top
    c0 = left
    r1 = top + height - 1
    c1 = left + width - 1
    for other_top, other_left, other_height, other_width in rects:
        rr0 = other_top
        cc0 = other_left
        rr1 = other_top + other_height - 1
        cc1 = other_left + other_width - 1
        if not (r1 + 1 < rr0 or rr1 + 1 < r0 or c1 + 1 < cc0 or cc1 + 1 < c0):
            return False
    return True


def _place_pair_rectangles(specs: list[dict]) -> list[dict] | None:
    work_h, work_w = WORK_SHAPE
    occupied_rects: list[tuple[int, int, int, int]] = []

    for spec in specs:
        left = spec["left"]
        right = spec["right"]
        lh, lw = map(int, left.shape)
        rh, rw = map(int, right.shape)

        max_left_col = work_w - lw - rw - PAIR_GAP_MIN
        if max_left_col < 0:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            left_top = random.randint(0, work_h - lh)
            left_col = random.randint(0, max_left_col)
            right_top = random.randint(0, work_h - rh)
            min_right_col = left_col + lw + PAIR_GAP_MIN
            if min_right_col > work_w - rw:
                continue
            right_col = random.randint(min_right_col, work_w - rw)

            if not _rectangles_separated(occupied_rects, left_top, left_col, lh, lw):
                continue
            if not _rectangles_separated(occupied_rects, right_top, right_col, rh, rw):
                continue

            occupied_rects.append((left_top, left_col, lh, lw))
            occupied_rects.append((right_top, right_col, rh, rw))
            spec["left_pos"] = (left_top, left_col)
            spec["right_pos"] = (right_top, right_col)
            placed = True
            break

        if not placed:
            return None
    return specs


def _paste(sprite: np.ndarray, grid: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    mask = sprite != BG
    grid[top : top + h, left : left + w][mask] = sprite[mask]


def _crop_with_random_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_pad = random.randint(0, min(EXTRA_MARGIN_MAX, int(r0)))
    left_pad = random.randint(0, min(EXTRA_MARGIN_MAX, int(c0)))
    bottom_pad = random.randint(0, min(EXTRA_MARGIN_MAX, grid.shape[0] - 1 - int(r1)))
    right_pad = random.randint(0, min(EXTRA_MARGIN_MAX, grid.shape[1] - 1 - int(c1)))
    return grid[r0 - top_pad : r1 + bottom_pad + 1, c0 - left_pad : c1 + right_pad + 1]


def _render_expected(specs: list[dict]) -> np.ndarray:
    ordered = sorted(specs, key=lambda spec: int(spec["height"]))
    out_h = max(int(spec["height"]) for spec in ordered)
    out_w = sum(int(spec["rect_width"]) for spec in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for spec in ordered:
        height = int(spec["height"])
        width = int(spec["rect_width"])
        color = int(spec["color"])
        out[out_h - height :, col : col + width] = color
        col += width + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_pair_specs()
    if specs is None:
        return None
    specs = _place_pair_rectangles(specs)
    if specs is None:
        return None

    work = np.full(WORK_SHAPE, BG, dtype=int)
    for spec in specs:
        _paste(spec["left"], work, *spec["left_pos"])
        _paste(spec["right"], work, *spec["right_pos"])

    inp = _crop_with_random_margin(work)
    expected = _render_expected(specs)
    if max(inp.shape) > ARC_MAX or max(expected.shape) > ARC_MAX:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
