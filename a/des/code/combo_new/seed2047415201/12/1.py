import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500


def _find_monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == bg or seen[sr, sc]:
                continue

            seen[sr, sc] = True
            stack = [(sr, sc)]
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize rectangles to vertical form, then nest them at the lower-left corner.

    Input structure:
    - Black background with several separated monochrome solid foreground rectangles.
    - Each rectangle has a distinct color and a non-square bounding box.
    - Every rectangle is either wider than tall or taller than wide, and areas are unique.

    Rule:
    - For each foreground rectangle, rotate it 90 degrees when its width is greater
      than its height; rectangles already at least as tall as wide keep their shape.
    - Sort the normalized rectangles by area from largest to smallest.
    - Redraw them on a same-size black canvas with all lower-left corners aligned to
      the grid lower-left corner, painting smaller rectangles over larger ones.

    Key insight:
    - Aspect ratio decides the orientation normalization, while area independently
      decides the nested drawing order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    objects = []
    for component in _find_monochrome_components(g, BG):
        r0, c0, r1, c1 = component["bbox"]
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        area = int(component["area"])
        color = int(component["color"])

        assert area == obj_h * obj_w
        assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color))

        if obj_w > obj_h:
            out_h, out_w = obj_w, obj_h
        else:
            out_h, out_w = obj_h, obj_w
        assert out_h <= height and out_w <= width
        objects.append((area, color, out_h, out_w))

    assert objects
    areas = [area for area, _color, _out_h, _out_w in objects]
    assert len(areas) == len(set(areas))

    out = np.full_like(g, BG)
    for _area, color, obj_h, obj_w in sorted(objects, key=lambda item: item[0], reverse=True):
        out[height - obj_h : height, 0:obj_w] = color
    return out


def _sample_nested_dimensions(count: int, cap: int) -> list[tuple[int, int]] | None:
    short_min = 2
    short_max = min(8, cap - 1)
    if short_max - short_min + 1 < count:
        return None

    widths = sorted(random.sample(range(short_min, short_max + 1), count), reverse=True)
    slack = cap - widths[0]
    if slack < 1:
        return None

    extras = sorted([random.randint(1, slack) for _ in range(count)], reverse=True)
    dims = [(w + extra, w) for w, extra in zip(widths, extras)]
    if any(h <= w for h, w in dims):
        return None
    return dims


def _sample_orientations(count: int) -> list[bool]:
    wide_count = count // 2
    if count % 2 and random.random() < 0.5:
        wide_count += 1
    wide_count = min(max(wide_count, 1), count - 1)
    orientations = [True] * wide_count + [False] * (count - wide_count)
    random.shuffle(orientations)
    return orientations


def _can_place(occupied: np.ndarray, row: int, col: int, h: int, w: int) -> bool:
    height, width = occupied.shape
    r0 = max(0, row - 1)
    c0 = max(0, col - 1)
    r1 = min(height, row + h + 1)
    c1 = min(width, col + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_rectangles(shape: tuple[int, int], specs: list[dict]) -> np.ndarray | None:
    height, width = shape
    grid = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["src_h"] * specs[i]["src_w"], reverse=True)

    for i in order:
        spec = specs[i]
        src_h, src_w = int(spec["src_h"]), int(spec["src_w"])
        candidates = [
            (r, c)
            for r in range(height - src_h + 1)
            for c in range(width - src_w + 1)
            if _can_place(occupied, r, c, src_h, src_w)
        ]
        if not candidates:
            return None
        row, col = random.choice(candidates)
        grid[row : row + src_h, col : col + src_w] = int(spec["color"])
        occupied[row : row + src_h, col : col + src_w] = True

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 30)
    width = random.randint(14, 30)
    count = random.randint(3, 5)
    cap = min(height, width) - 2

    normalized_dims = _sample_nested_dimensions(count, cap)
    if normalized_dims is None:
        return None

    orientations = _sample_orientations(count)
    colors = random.sample(COLORS, count)
    specs = []
    for (norm_h, norm_w), is_wide, color in zip(normalized_dims, orientations, colors):
        if is_wide:
            src_h, src_w = norm_w, norm_h
        else:
            src_h, src_w = norm_h, norm_w
        specs.append(
            {
                "color": int(color),
                "src_h": int(src_h),
                "src_w": int(src_w),
                "norm_h": int(norm_h),
                "norm_w": int(norm_w),
                "area": int(norm_h * norm_w),
            }
        )

    areas = [spec["area"] for spec in specs]
    if len(areas) != len(set(areas)):
        return None

    grid = _place_rectangles((height, width), specs)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if len(_find_monochrome_components(grid, BG)) != count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
