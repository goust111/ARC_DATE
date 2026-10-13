from __future__ import annotations

import random
from collections import deque

import numpy as np

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
PLACE_RETRIES = 120
COLOR_IDS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[Coord] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame(grid: np.ndarray, coords: list[Coord], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if len(coords) != 2 * (height + width) - 4:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    perimeter = np.zeros((height, width), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    if not np.all(sub[perimeter] == color):
        return False
    if np.any(sub[~perimeter] == color):
        return False
    return True


def _template_relation(frame_bbox: BBox, coords: list[Coord]) -> tuple[Coord, Coord] | None:
    if len(coords) != 2:
        return None
    (r0, c0), (r1, c1) = sorted(coords)
    fr0, fc0, fr1, fc1 = frame_bbox

    if r0 == r1 and abs(c0 - c1) == 1:
        row = r0
        lo, hi = sorted((c0, c1))
        if lo == fc0 - 2 and hi == fc0 - 1 and fr0 <= row <= fr1:
            return (row, fc0 - 1), (row, fc0 - 2)
        if lo == fc1 + 1 and hi == fc1 + 2 and fr0 <= row <= fr1:
            return (row, fc1 + 1), (row, fc1 + 2)

    if c0 == c1 and abs(r0 - r1) == 1:
        col = c0
        lo, hi = sorted((r0, r1))
        if lo == fr0 - 2 and hi == fr0 - 1 and fc0 <= col <= fc1:
            return (fr0 - 1, col), (fr0 - 2, col)
        if lo == fr1 + 1 and hi == fr1 + 2 and fc0 <= col <= fc1:
            return (fr1 + 1, col), (fr1 + 2, col)

    return None


def _extract_layouts(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            if _is_frame(grid, coords, color):
                frames.append(
                    {
                        "bbox": _bbox(coords),
                        "color": color,
                        "coords": coords,
                    }
                )

    assert frames, "expected at least one hollow rectangular frame"
    frames.sort(key=lambda frame: frame["bbox"])

    frame_mask = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        for r, c in frame["coords"]:
            frame_mask[r, c] = True

    other_components = _components((grid != bg) & (~frame_mask))
    template_components = [coords for coords in other_components if len(coords) == 2]

    layouts: list[dict[str, object]] = []
    for frame in frames:
        frame_bbox = frame["bbox"]
        matches: list[tuple[Coord, Coord]] = []
        for coords in template_components:
            relation = _template_relation(frame_bbox, coords)
            if relation is not None:
                matches.append(relation)
        assert len(matches) == 1, "each frame must have exactly one adjacent two-cell template"

        near, far = matches[0]
        outer_color = int(grid[near])
        inner_color = int(grid[far])
        assert outer_color != inner_color, "template colors must differ"
        assert outer_color != bg and inner_color != bg, "template colors must be foreground"

        r0, c0, r1, c1 = frame_bbox
        seeds = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) != bg
        ]
        assert seeds, "each frame must contain at least one seed"
        layouts.append(
            {
                "bbox": frame_bbox,
                "outer_color": outer_color,
                "inner_color": inner_color,
                "seeds": seeds,
            }
        )

    return layouts


def _paint_layouts(grid: np.ndarray, layouts: list[dict[str, object]], bg: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    for layout in layouts:
        r0, c0, r1, c1 = layout["bbox"]
        outer_color = int(layout["outer_color"])
        inner_color = int(layout["inner_color"])
        seeds = list(layout["seeds"])

        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        local_h = r1 - r0 - 1
        local_w = c1 - c0 - 1
        dist = np.full((local_h, local_w), local_h + local_w + 1, dtype=int)

        for sr, sc in seeds:
            lr = sr - r0 - 1
            lc = sc - c0 - 1
            for c in range(c0 + 1, c1):
                dist[lr, c - c0 - 1] = min(dist[lr, c - c0 - 1], abs(c - sc))
            for r in range(r0 + 1, r1):
                dist[r - r0 - 1, lc] = min(dist[r - r0 - 1, lc], abs(r - sr))

        sub = out[r0 + 1 : r1, c0 + 1 : c1]
        covered = dist <= local_h + local_w
        sub[covered] = np.where(dist[covered] % 2 == 0, inner_color, outer_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed-centered crosses inside hollow frames, then recolor each cross
    by the parity encoded in the adjacent two-cell template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    layouts = _extract_layouts(g, bg)
    return _paint_layouts(g, layouts, bg)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _structure_box(r0: int, c0: int, height: int, width: int, side: str) -> BBox:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if side == "top":
        return r0 - 2, c0, r1, c1
    if side == "bottom":
        return r0, c0, r1 + 2, c1
    if side == "left":
        return r0, c0 - 2, r1, c1
    assert side == "right"
    return r0, c0, r1, c1 + 2


def _sample_seed_positions(r0: int, c0: int, height: int, width: int, count: int) -> list[Coord]:
    rows = random.sample(range(r0 + 1, r0 + height - 1), count)
    cols = random.sample(range(c0 + 1, c0 + width - 1), count)
    random.shuffle(cols)
    return list(zip(rows, cols))


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _template_cells(bbox: BBox, side: str, offset: int) -> tuple[Coord, Coord]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return (r0 - 1, c0 + offset), (r0 - 2, c0 + offset)
    if side == "bottom":
        return (r1 + 1, c0 + offset), (r1 + 2, c0 + offset)
    if side == "left":
        return (r0 + offset, c0 - 1), (r0 + offset, c0 - 2)
    assert side == "right"
    return (r0 + offset, c1 + 1), (r0 + offset, c1 + 2)


def _place_one_structure(
    grid_shape: tuple[int, int],
    reserved: list[BBox],
    bg: int,
    frame_color: int,
    target_count: int,
) -> tuple[dict[str, object], list[tuple[int, int, int]], BBox] | None:
    h, w = grid_shape
    max_dim = 7 if target_count >= 3 else 8 if target_count == 2 else 10

    for _ in range(PLACE_RETRIES):
        side = random.choice(SIDES)
        max_h = min(max_dim, h - (2 if side in ("top", "bottom") else 0))
        max_w = min(max_dim, w - (2 if side in ("left", "right") else 0))
        if max_h < 5 or max_w < 5:
            return None

        height = random.randint(5, max_h)
        width = random.randint(5, max_w)

        min_r0 = 2 if side == "top" else 0
        max_r0 = h - height - (2 if side == "bottom" else 0)
        min_c0 = 2 if side == "left" else 0
        max_c0 = w - width - (2 if side == "right" else 0)
        if max_r0 < min_r0 or max_c0 < min_c0:
            continue

        r0 = random.randint(min_r0, max_r0)
        c0 = random.randint(min_c0, max_c0)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        padded_box = _expand_box(_structure_box(r0, c0, height, width, side), 1, grid_shape)
        if any(_boxes_overlap(padded_box, other) for other in reserved):
            continue

        max_seeds = min(3, height - 2, width - 2)
        seed_count = random.randint(1, max_seeds)
        seeds = _sample_seed_positions(r0, c0, height, width, seed_count)

        remaining = [color for color in COLOR_IDS if color not in {bg, frame_color}]
        outer_color, inner_color = random.sample(remaining, 2)
        seed_palette = [color for color in remaining if color not in {outer_color, inner_color}]
        seed_colors = random.sample(seed_palette, seed_count)

        offset = random.randint(0, width - 1) if side in ("top", "bottom") else random.randint(0, height - 1)
        near, far = _template_cells(bbox, side, offset)

        paints: list[tuple[int, int, int]] = []
        r1, c1 = bbox[2], bbox[3]
        for c in range(c0, c1 + 1):
            paints.append((r0, c, frame_color))
            paints.append((r1, c, frame_color))
        for r in range(r0 + 1, r1):
            paints.append((r, c0, frame_color))
            paints.append((r, c1, frame_color))
        paints.append((near[0], near[1], outer_color))
        paints.append((far[0], far[1], inner_color))
        for (sr, sc), seed_color in zip(seeds, seed_colors):
            paints.append((sr, sc, seed_color))

        layout = {
            "bbox": bbox,
            "outer_color": outer_color,
            "inner_color": inner_color,
            "seeds": seeds,
        }
        return layout, paints, padded_box

    return None


def _generate_once() -> np.ndarray | None:
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    bg, frame_color = random.sample(COLOR_IDS, 2)
    grid = np.full((height, width), bg, dtype=int)

    if min(height, width) <= 19:
        target_count = random.choice((1, 2))
    else:
        target_count = random.choice((1, 2, 3))

    layouts: list[dict[str, object]] = []
    reserved: list[BBox] = []
    for _ in range(target_count):
        placed = _place_one_structure(grid.shape, reserved, bg, frame_color, target_count)
        if placed is None:
            return None
        layout, paints, padded_box = placed
        for r, c, color in paints:
            grid[r, c] = color
        layouts.append(layout)
        reserved.append(padded_box)

    expected = _paint_layouts(grid, layouts, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
