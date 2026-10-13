import math
import random

import numpy as np


BG = 0
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_rectangles(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rectangles = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            area = rect_h * rect_w

            assert len(coords) == area, "each object must be a solid rectangle"
            assert rect_h % 2 == 1 and rect_w % 2 == 1, "rectangle dimensions must be odd"
            assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)), "rectangle bbox must be filled"

            rectangles.append(
                {
                    "color": color,
                    "top": r0,
                    "left": c0,
                    "height": rect_h,
                    "width": rect_w,
                    "area": area,
                    "center_r": (r0 + r1) // 2,
                    "center_c": (c0 + c1) // 2,
                }
            )

    assert rectangles, "expected at least one rectangle"
    areas = [int(rect["area"]) for rect in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must be unique"
    return rectangles


def _draw_centered(out: np.ndarray, center_r: int, center_c: int, rect_h: int, rect_w: int, color: int) -> None:
    r0 = center_r - rect_h // 2
    c0 = center_c - rect_w // 2
    r1 = r0 + rect_h
    c1 = c0 + rect_w
    assert 0 <= r0 and r1 <= out.shape[0]
    assert 0 <= c0 and c1 <= out.shape[1]
    out[r0:r1, c0:c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack scattered solid rectangles onto the largest rectangle's center to form concentric nested frames.

    Input structure:
    - Black background.
    - Several disconnected monochrome solid rectangles are scattered across the grid.
    - Each rectangle has a distinct color, a unique area, and odd width and height.

    Rule:
    - Find every solid rectangle and compute its area and center.
    - Use the largest rectangle's center as the common stacking anchor.
    - Clear the whole canvas to Black.
    - Sort rectangles by area from largest to smallest, then redraw each one at the anchor with its original size and color.
    - Later, smaller rectangles overwrite the centers of earlier, larger ones, leaving visible colored border rings.

    Key insight:
    - The transformation is a global relocation to one shared center, and the area order determines the visible nesting of the concentric layers.
    """
    g = np.asarray(grid, dtype=int)
    rectangles = sorted(_extract_rectangles(g), key=lambda rect: int(rect["area"]), reverse=True)

    anchor_r = int(rectangles[0]["center_r"])
    anchor_c = int(rectangles[0]["center_c"])

    out = np.zeros_like(g)
    for rect in rectangles:
        _draw_centered(
            out,
            anchor_r,
            anchor_c,
            int(rect["height"]),
            int(rect["width"]),
            int(rect["color"]),
        )
    return out


def _random_odd(rng: random.Random, low: int, high: int):
    values = [v for v in range(low, high + 1) if v % 2 == 1]
    if not values:
        return None
    return rng.choice(values)


def _sample_sizes(rng: random.Random, count: int):
    min_outer = 2 * count + 1
    mode = rng.choice(("wide", "tall", "balanced"))

    for _ in range(80):
        if mode == "wide":
            outer_h = _random_odd(rng, min_outer, 17)
            if outer_h is None:
                continue
            outer_w = _random_odd(rng, max(min_outer, outer_h + 4), 25)
        elif mode == "tall":
            outer_w = _random_odd(rng, min_outer, 17)
            if outer_w is None:
                continue
            outer_h = _random_odd(rng, max(min_outer, outer_w + 4), 25)
        else:
            outer_h = _random_odd(rng, min_outer, 21)
            if outer_h is None:
                continue
            delta = rng.choice((0, 2, 4))
            outer_w = _random_odd(rng, max(min_outer, outer_h - delta), min(21, outer_h + delta))

        if outer_h is None or outer_w is None:
            continue

        height_choices = list(range(3, outer_h + 1, 2))
        width_choices = list(range(3, outer_w + 1, 2))
        if len(height_choices) < count or len(width_choices) < count:
            continue

        heights = sorted(rng.sample(height_choices, count), reverse=True)
        widths = sorted(rng.sample(width_choices, count), reverse=True)
        areas = [h * w for h, w in zip(heights, widths)]
        if len(set(areas)) != count:
            continue

        sizes = list(zip(heights, widths))
        return mode, sizes

    return None


def _sample_canvas(rng: random.Random, sizes, mode: str):
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    padded_area = sum((h + 2) * (w + 2) for h, w in sizes)
    target_side = int(math.ceil(math.sqrt(padded_area)))

    for _ in range(40):
        if mode == "wide":
            min_h = max(max_h + 2, target_side - rng.randint(3, 6))
            min_w = max(max_w + 2, target_side + rng.randint(0, 4))
        elif mode == "tall":
            min_h = max(max_h + 2, target_side + rng.randint(0, 4))
            min_w = max(max_w + 2, target_side - rng.randint(3, 6))
        else:
            slack = rng.randint(0, 3)
            min_h = max(max_h + 2, target_side - slack)
            min_w = max(max_w + 2, target_side - slack)

        min_h = max(13, min_h)
        min_w = max(13, min_w)
        if min_h > 30 or min_w > 30:
            continue

        height = rng.randint(min_h, 30)
        width = rng.randint(min_w, 30)
        if height * width < padded_area:
            continue
        return height, width

    return None


def _place_rectangles(rng: random.Random, height: int, width: int, sizes):
    blocked = np.zeros((height, width), dtype=bool)
    placements = []

    for rect_h, rect_w in sizes:
        positions = []
        for r0 in range(height - rect_h + 1):
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + rect_h + 1)
            for c0 in range(width - rect_w + 1):
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c0 + rect_w + 1)
                if not bool(blocked[rr0:rr1, cc0:cc1].any()):
                    positions.append((r0, c0))

        if not positions:
            return None

        r0, c0 = rng.choice(positions)
        placements.append((r0, c0))
        blocked[max(0, r0 - 1) : min(height, r0 + rect_h + 1), max(0, c0 - 1) : min(width, c0 + rect_w + 1)] = True

    return placements


def _gen_attempt(rng: random.Random):
    count = rng.randint(3, 5)
    sampled = _sample_sizes(rng, count)
    if sampled is None:
        return None

    mode, sizes = sampled
    for _ in range(40):
        canvas = _sample_canvas(rng, sizes, mode)
        if canvas is None:
            continue
        height, width = canvas
        placements = _place_rectangles(rng, height, width, sizes)
        if placements is None:
            continue

        colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], count)
        grid = np.zeros((height, width), dtype=int)
        for (rect_h, rect_w), (r0, c0), color in zip(sizes, placements, colors):
            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color

        rectangles = _extract_rectangles(grid)
        if len(rectangles) != count:
            continue
        if np.array_equal(main(grid), grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
