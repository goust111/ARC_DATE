import random
from dataclasses import dataclass

import numpy as np


BG = 0
ARC_MAX = 30
GEN_RETRIES = 400
TOP_H_RANGE = (2, 5)
TOP_W_RANGE = (2, 4)
MARGIN_RANGE = (1, 2)
STACK_GAP_RANGE = (2, 4)
SIDE_PAD_RANGE = (1, 3)
HEIGHT_EXTRA_RANGE = (4, 8)
STACK_COUNT_RANGE = (2, 3)
RECT_COUNT_RANGE = (3, 5)


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    h: int
    w: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    @property
    def area(self) -> int:
        return self.h * self.w


def _paint_rect(grid: np.ndarray, rect: Rect) -> None:
    grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = rect.color


def _render(rects: list[Rect], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for rect in rects:
        _paint_rect(out, rect)
    return out


def _color_specs(grid: np.ndarray) -> list[dict[str, int]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    assert colors, "expected at least one non-background rectangle"
    specs = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        assert mask[0].any() and mask[-1].any(), "visible pixels must reach the true top and bottom"
        assert mask[:, 0].any() and mask[:, -1].any(), "visible pixels must reach the true left and right"
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        specs.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "area": h * w,
            }
        )
    return specs


def _pack_rectangles(specs: list[dict[str, int]]) -> np.ndarray:
    order = sorted(specs, key=lambda item: (-item["area"], -item["h"], -item["w"], item["r0"], item["c0"]))
    out_h = max(item["h"] for item in order)
    out_w = sum(item["w"] for item in order) + len(order) - 1
    assert out_h <= ARC_MAX and out_w <= ARC_MAX
    out = np.full((out_h, out_w), BG, dtype=int)
    c = 0
    for item in order:
        h = item["h"]
        w = item["w"]
        out[out_h - h :, c : c + w] = item["color"]
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the occluded rectangles and repack them as a bottom-aligned row sorted by area.

    Input structure:
    - The background is black.
    - Each non-black color belongs to exactly one axis-aligned rectangle.
    - Higher rectangles may cover lower ones, so one color can survive only as boundary fragments such as L-shapes, U-shapes, or paired strips.

    Rule:
    - For each non-black color, gather all visible pixels of that color.
    - The surviving fragments still touch the rectangle's true top, bottom, left, and right sides, so the full rectangle is the bounding box of all visible pixels of that color.
    - Compute each rectangle's area, sort rectangles from largest to smallest area, and place them left-to-right on a black output canvas with one black column between neighbors and all rectangles bottom-aligned.

    Key insight:
    - Occlusion breaks one rectangle into several visible pieces, but the color stays unique, so reunifying all pixels of one color recovers the hidden rectangle before the final area-based sort.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _pack_rectangles(_color_specs(g))


def _sample_lengths(total: int, rng: random.Random) -> list[int]:
    for _ in range(100):
        n_stacks = rng.randint(STACK_COUNT_RANGE[0], min(STACK_COUNT_RANGE[1], total))
        lengths = [1] * n_stacks
        remain = total - n_stacks
        while remain > 0:
            idx = rng.randrange(n_stacks)
            if lengths[idx] < 3:
                lengths[idx] += 1
                remain -= 1
        if max(lengths) > 1:
            rng.shuffle(lengths)
            return lengths
    raise RuntimeError("failed to sample stack lengths")


def _touch_mode(rng: random.Random) -> tuple[int, int, int, int]:
    modes = ("frame", "one_side", "adjacent_two", "opposite_two")
    mode = rng.choice(modes)
    top = bottom = left = right = 1
    if mode == "frame":
        pass
    elif mode == "one_side":
        zero = rng.choice(("top", "bottom", "left", "right"))
        if zero == "top":
            top = 0
        elif zero == "bottom":
            bottom = 0
        elif zero == "left":
            left = 0
        else:
            right = 0
    elif mode == "adjacent_two":
        zeros = rng.choice((("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")))
        for side in zeros:
            if side == "top":
                top = 0
            elif side == "bottom":
                bottom = 0
            elif side == "left":
                left = 0
            else:
                right = 0
    else:
        zeros = rng.choice((("top", "bottom"), ("left", "right")))
        for side in zeros:
            if side == "top":
                top = 0
            elif side == "bottom":
                bottom = 0
            elif side == "left":
                left = 0
            else:
                right = 0

    def grow(v: int) -> int:
        return v * rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])

    return grow(top), grow(bottom), grow(left), grow(right)


def _build_stack(length: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    rects = [(0, 0, rng.randint(*TOP_H_RANGE), rng.randint(*TOP_W_RANGE))]
    for _ in range(length - 1):
        child_r0, child_c0, child_h, child_w = rects[0]
        top, bottom, left, right = _touch_mode(rng)
        parent_h = child_h + top + bottom
        parent_w = child_w + left + right
        rects = [(r0 + top, c0 + left, h, w) for r0, c0, h, w in rects]
        rects.insert(0, (0, 0, parent_h, parent_w))
    return rects


def _true_specs(rects: list[Rect]) -> list[dict[str, int]]:
    return [
        {
            "color": rect.color,
            "r0": rect.r0,
            "c0": rect.c0,
            "h": rect.h,
            "w": rect.w,
            "area": rect.area,
        }
        for rect in rects
    ]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    total_rects = rng.randint(*RECT_COUNT_RANGE)
    lengths = _sample_lengths(total_rects, rng)
    local_stacks = [_build_stack(length, rng) for length in lengths]

    root_sizes = [(stack[0][2], stack[0][3]) for stack in local_stacks]
    gaps = [rng.randint(*STACK_GAP_RANGE) for _ in range(max(0, len(local_stacks) - 1))]
    left_pad = rng.randint(*SIDE_PAD_RANGE)
    right_pad = rng.randint(*SIDE_PAD_RANGE)
    total_width = left_pad + right_pad + sum(w for _h, w in root_sizes) + sum(gaps)
    if total_width > ARC_MAX:
        return None

    max_root_h = max(h for h, _w in root_sizes)
    height = max_root_h + rng.randint(*HEIGHT_EXTRA_RANGE)
    if height > ARC_MAX:
        return None

    colors = rng.sample(list(range(1, 10)), total_rects)
    order = list(range(len(local_stacks)))
    rng.shuffle(order)

    rects: list[Rect] = []
    x = left_pad
    color_idx = 0
    for pos, stack_idx in enumerate(order):
        local_rects = local_stacks[stack_idx]
        root_h = local_rects[0][2]
        root_w = local_rects[0][3]
        y = rng.randint(1, height - root_h - 1)
        for r0, c0, h, w in local_rects:
            rects.append(Rect(colors[color_idx], y + r0, x + c0, h, w))
            color_idx += 1
        if pos < len(order) - 1:
            x += root_w + gaps[pos]

    areas = [rect.area for rect in rects]
    if len(set(areas)) != len(areas):
        return None

    packed_width = sum(rect.w for rect in rects) + len(rects) - 1
    if packed_width > ARC_MAX:
        return None

    grid = _render(rects, (height, total_width))
    if not any(np.count_nonzero(grid == rect.color) < rect.area for rect in rects):
        return None

    for rect in rects:
        visible = np.argwhere(grid == rect.color)
        if visible.size == 0:
            return None
        if visible[:, 0].min() != rect.r0 or visible[:, 0].max() != rect.r1:
            return None
        if visible[:, 1].min() != rect.c0 or visible[:, 1].max() != rect.c1:
            return None

    position_order = [rect.color for rect in sorted(rects, key=lambda item: (item.c0, item.r0))]
    area_order = [rect.color for rect in sorted(rects, key=lambda item: (-item.area, -item.h, -item.w, item.r0, item.c0))]
    if position_order == area_order:
        return None

    expected = _pack_rectangles(_true_specs(rects))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
