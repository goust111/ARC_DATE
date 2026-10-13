from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400

ITEM_COUNT_RANGE = (3, 5)
OUTER_HEIGHT_RANGE = (6, 12)
OUTER_WIDTH_RANGE = (6, 12)
MIN_SIDE = 2
MAX_SIDE = 14
MAX_CLIP_OVERSHOOT = 4

ROW_GAP_RANGE = (1, 3)
COL_GAP_RANGE = (1, 3)
SCENE_X_RANGE = (0, 4)
SCENE_Y_RANGE = (0, 3)
BOTTOM_GAP_RANGE = (1, 3)
RIGHT_PAD_RANGE = (0, 6)
BOTTOM_PAD_RANGE = (0, 2)
LEGEND_EXTRA_GAP_OPTIONS = (0, 1)
LEGEND_EXTRA_SLACK_RANGE = (0, 5)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _legend_colors(grid: np.ndarray) -> list[int]:
    colors = [int(value) for value in grid[-1] if int(value) != BG]
    assert len(colors) >= 2, "legend must contain at least two colors"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return colors


def _rectangle_dims(scene: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(scene == int(color))
    assert coords.size > 0, "each legend color must appear in the scene"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = scene[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch == int(color)), "each object must be a solid rectangle"
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _render_nested(rectangles: list[tuple[int, int, int]]) -> np.ndarray:
    outer_color, outer_h, outer_w = rectangles[0]
    out = np.zeros((int(outer_h), int(outer_w)), dtype=int)
    out[:, :] = int(outer_color)

    region_r, region_c = 0, 0
    region_h, region_w = int(outer_h), int(outer_w)
    for color, h, w in rectangles[1:]:
        fit_h = min(int(h), int(region_h))
        fit_w = min(int(w), int(region_w))
        top = int(region_r + (region_h - fit_h) // 2)
        left = int(region_c + (region_w - fit_w) // 2)
        out[top : top + fit_h, left : left + fit_w] = int(color)
        region_r, region_c = top, left
        region_h, region_w = fit_h, fit_w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the scattered rectangles by legend order and redraw them as
    centered nested solid rectangles. The leftmost legend color becomes the
    outermost layer; later layers are clipped to the previous layer's usable
    size when necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a legend row"

    legend = _legend_colors(g)
    scene = g[:-1]
    scene_colors = set(map(int, np.unique(scene))) - {BG}
    assert scene_colors == set(legend), "scene colors must match the legend colors"

    rectangles = [(color, *_rectangle_dims(scene, color)) for color in legend]
    return _render_nested(rectangles)


def _sample_fit_dims(max_h: int, max_w: int, used_areas: set[int]) -> tuple[int, int] | None:
    candidates = [
        (h, w)
        for h in range(MIN_SIDE, int(max_h) + 1)
        for w in range(MIN_SIDE, int(max_w) + 1)
        if h * w not in used_areas
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_clip_dims(current_h: int, current_w: int, used_areas: set[int]) -> tuple[int, int] | None:
    max_h = min(MAX_SIDE, int(current_h) + MAX_CLIP_OVERSHOOT)
    max_w = min(MAX_SIDE, int(current_w) + MAX_CLIP_OVERSHOOT)
    current_area = int(current_h) * int(current_w)
    candidates: list[tuple[int, int]] = []

    if current_w > MIN_SIDE:
        for h in range(int(current_h) + 1, int(max_h) + 1):
            for w in range(MIN_SIDE, int(current_w)):
                area = h * w
                if area > current_area and area not in used_areas:
                    candidates.append((h, w))

    if current_h > MIN_SIDE:
        for h in range(MIN_SIDE, int(current_h)):
            for w in range(int(current_w) + 1, int(max_w) + 1):
                area = h * w
                if area > current_area and area not in used_areas:
                    candidates.append((h, w))

    if not candidates:
        return None
    return random.choice(candidates)


def _sample_dimension_sequence(n_items: int) -> tuple[list[tuple[int, int]], bool] | None:
    for _ in range(200):
        need_clip = random.choice([False, True])
        clip_index = random.randint(1, min(n_items - 1, 2)) if need_clip else -1

        dims = [(random.randint(*OUTER_HEIGHT_RANGE), random.randint(*OUTER_WIDTH_RANGE))]
        used_areas = {dims[0][0] * dims[0][1]}
        current_h, current_w = dims[0]
        ok = True

        for idx in range(1, n_items):
            if idx == clip_index:
                picked = _sample_clip_dims(current_h, current_w, used_areas)
                if picked is None:
                    ok = False
                    break
                h, w = picked
                current_h = min(current_h, h)
                current_w = min(current_w, w)
            else:
                picked = _sample_fit_dims(current_h, current_w, used_areas)
                if picked is None:
                    ok = False
                    break
                h, w = picked
                current_h, current_w = h, w
            dims.append((int(h), int(w)))
            used_areas.add(int(h) * int(w))

        if not ok:
            continue

        areas = [h * w for h, w in dims]
        if need_clip and areas == sorted(areas, reverse=True):
            continue
        return dims, need_clip
    return None


def _partition_sizes(n: int, k: int) -> list[int]:
    if k == 1:
        return [n]
    cuts = sorted(random.sample(range(1, n), k - 1))
    parts = []
    prev = 0
    for cut in cuts + [n]:
        parts.append(cut - prev)
        prev = cut
    return parts


def _sample_scene_layout(rectangles: list[tuple[int, int, int]]) -> tuple[int, int, list[tuple[int, int, int, int, int]]]:
    shuffled = rectangles[:]
    random.shuffle(shuffled)

    row_cap = min(3, len(shuffled))
    row_floor = 1 if len(shuffled) <= 3 else 2
    row_count = random.randint(row_floor, row_cap)
    row_sizes = _partition_sizes(len(shuffled), row_count)

    rows: list[dict] = []
    index = 0
    for size in row_sizes:
        items = shuffled[index : index + size]
        index += size
        gaps = [random.randint(*COL_GAP_RANGE) for _ in range(max(0, size - 1))]
        row_h = max(h for _color, h, _w in items)
        row_w = sum(w for _color, _h, w in items) + sum(gaps)
        rows.append({"items": items, "gaps": gaps, "height": row_h, "width": row_w})

    scene_w = max(int(row["width"]) for row in rows)
    row_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(max(0, row_count - 1))]
    scene_h = sum(int(row["height"]) for row in rows) + sum(row_gaps)

    placements: list[tuple[int, int, int, int, int]] = []
    y = 0
    for row_idx, row in enumerate(rows):
        row_h = int(row["height"])
        x = random.randint(0, int(scene_w - row["width"]))
        cursor = int(x)
        for item_idx, (color, h, w) in enumerate(row["items"]):
            dy = random.randint(0, row_h - int(h))
            placements.append((int(color), int(y + dy), int(cursor), int(h), int(w)))
            cursor += int(w)
            if item_idx < len(row["gaps"]):
                cursor += int(row["gaps"][item_idx])
        y += row_h
        if row_idx < len(row_gaps):
            y += int(row_gaps[row_idx])
    return int(scene_h), int(scene_w), placements


def _sample_spaced_positions(width: int, count: int, extra_gap: int) -> list[int]:
    usable = width - extra_gap * (count - 1)
    base = sorted(random.sample(range(usable), count))
    return [int(base[i] + i * extra_gap) for i in range(count)]


def _gen_attempt() -> np.ndarray | None:
    n_items = random.randint(*ITEM_COUNT_RANGE)
    colors = random.sample(PALETTE, n_items)
    legend = colors[:]
    random.shuffle(legend)

    dims_pack = _sample_dimension_sequence(n_items)
    if dims_pack is None:
        return None
    dims_by_order, need_clip = dims_pack

    rectangles = [(legend[i], dims_by_order[i][0], dims_by_order[i][1]) for i in range(n_items)]
    scene_h, scene_w, placements = _sample_scene_layout(rectangles)

    legend_gap = random.choice(LEGEND_EXTRA_GAP_OPTIONS)
    legend_needed = n_items + legend_gap * (n_items - 1)
    scene_x = random.randint(*SCENE_X_RANGE)
    scene_y = random.randint(*SCENE_Y_RANGE)
    right_pad = random.randint(*RIGHT_PAD_RANGE)
    bottom_gap = random.randint(*BOTTOM_GAP_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    extra_legend_slack = random.randint(*LEGEND_EXTRA_SLACK_RANGE)

    width = max(scene_x + scene_w + right_pad, legend_needed + extra_legend_slack)
    height = scene_y + scene_h + bottom_gap + 1 + bottom_pad

    grid = np.zeros((height, width), dtype=int)
    for color, r, c, h, w in placements:
        rr = scene_y + int(r)
        cc = scene_x + int(c)
        grid[rr : rr + int(h), cc : cc + int(w)] = int(color)

    legend_positions = _sample_spaced_positions(width, n_items, legend_gap)
    for col, color in zip(legend_positions, legend):
        grid[-1, int(col)] = int(color)

    areas = [h * w for _color, h, w in rectangles]
    if len(set(areas)) != n_items:
        return None
    if need_clip and areas == sorted(areas, reverse=True):
        return None

    expected = _render_nested(rectangles)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
