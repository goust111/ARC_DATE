from __future__ import annotations

import random
from collections import Counter

import numpy as np


MAX_GRID_SIZE = 30
GENERATION_ATTEMPTS = 500
ALL_COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _frame_info(grid: np.ndarray, bg: int, coords: np.ndarray) -> dict | None:
    color = int(grid[int(coords[0, 0]), int(coords[0, 1])])
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    rel = coords - np.array([r0, c0], dtype=int)
    if not np.all(border[rel[:, 0], rel[:, 1]]):
        return None
    if np.any(sub[~border] != int(bg)):
        return None

    border_size = int(border.sum())
    is_closed = int(coords.shape[0]) == border_size and bool(np.all(sub[border] == int(color)))
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "height": h,
        "width": w,
        "outer_area": h * w,
        "interior_area": (h - 2) * (w - 2),
        "closed": is_closed,
    }


def _closed_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = []
    for coords in _find_components(g, bg):
        info = _frame_info(g, bg, coords)
        if info is not None and info["closed"]:
            frames.append(info)
    return frames


def _has_outer_area_inversion(frames: list[dict]) -> bool:
    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            da = int(a["interior_area"]) - int(b["interior_area"])
            db = int(a["outer_area"]) - int(b["outer_area"])
            if da * db < 0:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed hollow frame with the color of the next larger interior area, wrapping the largest back to the smallest.

    Input structure:
    - The grid has a dominant background plus several one-cell-thick monochrome rectangular frame objects.
    - Some frames are closed rectangles with background-colored interiors; others have a gap on one side and stay open.
    - Frames do not overlap, do not nest, and closed-frame interior areas are all distinct.

    Rule:
    - Detect only the closed rectangular frames and ignore every open frame.
    - Sort the closed frames by the area strictly inside each border.
    - Fill the interior of each closed frame with the border color of the next larger closed frame; the largest interior wraps around to the smallest frame's color.

    Key insight:
    - The fill color is determined by a hidden successor relation in interior-area order, not by spatial proximity or by a frame's own color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _closed_frames(g)
    assert len(frames) >= 2, "expected at least two closed frames"

    areas = [int(frame["interior_area"]) for frame in frames]
    assert len(areas) == len(set(areas)), "closed-frame interior areas must be unique"

    ordered = sorted(frames, key=lambda frame: int(frame["interior_area"]))
    out = g.copy()
    for idx, frame in enumerate(ordered):
        next_color = int(ordered[(idx + 1) % len(ordered)]["color"])
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = next_color
    return out


def _frame_patch(
    h: int,
    w: int,
    color: int,
    bg: int,
    *,
    gap: tuple[str, int, int] | None = None,
) -> np.ndarray:
    patch = np.full((h, w), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    if gap is not None:
        side, start, length = gap
        if side == "top":
            patch[0, start : start + length] = int(bg)
        elif side == "bottom":
            patch[h - 1, start : start + length] = int(bg)
        elif side == "left":
            patch[start : start + length, 0] = int(bg)
        else:
            patch[start : start + length, w - 1] = int(bg)
    return patch


def _sample_gap(h: int, w: int) -> tuple[str, int, int]:
    vertical_ok = h > 3
    horizontal_ok = w > 3
    sides = []
    if horizontal_ok:
        sides.extend(["top", "bottom"])
    if vertical_ok:
        sides.extend(["left", "right"])
    side = random.choice(sides)
    if side in ("top", "bottom"):
        length = random.randint(1, min(3, w - 2))
        start = random.randint(1, w - 1 - length)
    else:
        length = random.randint(1, min(3, h - 2))
        start = random.randint(1, h - 1 - length)
    return side, start, length


def _sample_closed_dims(layout: str, count: int) -> list[tuple[int, int]]:
    for _ in range(300):
        if layout == "vertical":
            thin = (random.choice([3, 4]), random.randint(15, 24))
        else:
            thin = (random.randint(15, 24), random.choice([3, 4]))
        fat = (random.randint(6, 9), random.randint(6, 9))

        thin_inner = (thin[0] - 2) * (thin[1] - 2)
        fat_inner = (fat[0] - 2) * (fat[1] - 2)
        if thin_inner >= fat_inner or thin[0] * thin[1] <= fat[0] * fat[1]:
            continue

        dims = [thin, fat]
        used = {thin_inner, fat_inner}
        while len(dims) < count:
            if layout == "vertical":
                h = random.randint(3, 8)
                w = random.randint(4, 12)
            else:
                h = random.randint(4, 12)
                w = random.randint(3, 8)
            area = (h - 2) * (w - 2)
            if area in used:
                continue
            dims.append((h, w))
            used.add(area)

        random.shuffle(dims)
        temp = [
            {
                "height": h,
                "width": w,
                "interior_area": (h - 2) * (w - 2),
                "outer_area": h * w,
            }
            for h, w in dims
        ]
        if _has_outer_area_inversion(temp):
            return dims
    raise RuntimeError("failed to sample closed-frame dimensions")


def _sample_open_dims(layout: str, count: int) -> list[tuple[int, int]]:
    dims = []
    for _ in range(count):
        if layout == "vertical":
            dims.append((random.randint(3, 7), random.randint(4, 11)))
        else:
            dims.append((random.randint(4, 11), random.randint(3, 7)))
    return dims


def _place_objects(specs: list[dict], bg: int, layout: str) -> np.ndarray | None:
    main_sizes = [int(spec["height"] if layout == "vertical" else spec["width"]) for spec in specs]
    cross_sizes = [int(spec["width"] if layout == "vertical" else spec["height"]) for spec in specs]
    gaps = [random.randint(1, 3) for _ in range(max(0, len(specs) - 1))]

    main_used = sum(main_sizes) + sum(gaps)
    cross_used = max(cross_sizes)
    if main_used > MAX_GRID_SIZE or cross_used > MAX_GRID_SIZE:
        return None

    max_main_extra = min(6, MAX_GRID_SIZE - main_used)
    max_cross_extra = min(8, MAX_GRID_SIZE - cross_used)
    main_extent = main_used + random.randint(0, max_main_extra)
    cross_extent = cross_used + random.randint(0, max_cross_extra)

    total_fg = sum(2 * (spec["height"] + spec["width"]) - 4 for spec in specs)
    if main_extent * cross_extent - total_fg < max(12, max(2 * (spec["height"] + spec["width"]) - 4 for spec in specs)):
        return None

    grid_shape = (main_extent, cross_extent) if layout == "vertical" else (cross_extent, main_extent)
    grid = np.full(grid_shape, int(bg), dtype=int)

    pos = random.randint(0, main_extent - main_used)
    for idx, spec in enumerate(specs):
        h = int(spec["height"])
        w = int(spec["width"])
        if layout == "vertical":
            r0 = pos
            c0 = random.randint(0, cross_extent - w)
        else:
            r0 = random.randint(0, cross_extent - h)
            c0 = pos

        patch = _frame_patch(h, w, int(spec["color"]), int(bg), gap=spec.get("gap"))
        grid[r0 : r0 + h, c0 : c0 + w] = patch

        pos += main_sizes[idx]
        if idx < len(gaps):
            pos += gaps[idx]
    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(["vertical", "horizontal"])
    closed_count = random.choice([2, 3, 3, 4])
    open_count = random.choice([0, 1, 1, 2])

    closed_dims = _sample_closed_dims(layout, closed_count)
    open_dims = _sample_open_dims(layout, open_count)

    specs = [{"height": h, "width": w, "closed": True} for h, w in closed_dims]
    specs.extend({"height": h, "width": w, "closed": False} for h, w in open_dims)

    main_span = sum(spec["height"] if layout == "vertical" else spec["width"] for spec in specs)
    main_span += max(0, len(specs) - 1)
    cross_span = max(spec["width"] if layout == "vertical" else spec["height"] for spec in specs)
    if main_span > MAX_GRID_SIZE or cross_span > MAX_GRID_SIZE:
        return None

    bg = random.choice(ALL_COLORS)
    palette = [color for color in ALL_COLORS if color != bg]
    if len(specs) > len(palette):
        return None
    colors = random.sample(palette, len(specs))
    random.shuffle(specs)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)
        if not spec["closed"]:
            spec["gap"] = _sample_gap(int(spec["height"]), int(spec["width"]))

    grid = _place_objects(specs, bg, layout)
    if grid is None:
        return None

    frames = _closed_frames(grid)
    if len(frames) != closed_count:
        return None
    if not _has_outer_area_inversion(frames):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
