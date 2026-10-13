import random
from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    parts = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    assert boundary.size > 0, "expected a non-empty boundary"
    colors = np.unique(boundary)
    assert colors.size == 1, "each visible boundary must be monochrome"
    return int(colors[0])


def _count_top_rows(rect: np.ndarray, top: int, bottom: int, left: int, right: int, color: int) -> int:
    count = 0
    while top + count < bottom and np.all(rect[top + count, left:right] == color):
        count += 1
    return count


def _count_bottom_rows(rect: np.ndarray, top: int, bottom: int, left: int, right: int, color: int) -> int:
    count = 0
    while bottom - 1 - count >= top and np.all(rect[bottom - 1 - count, left:right] == color):
        count += 1
    return count


def _count_left_cols(rect: np.ndarray, top: int, bottom: int, left: int, right: int, color: int) -> int:
    count = 0
    while left + count < right and np.all(rect[top:bottom, left + count] == color):
        count += 1
    return count


def _count_right_cols(rect: np.ndarray, top: int, bottom: int, left: int, right: int, color: int) -> int:
    count = 0
    while right - 1 - count >= left and np.all(rect[top:bottom, right - 1 - count] == color):
        count += 1
    return count


def _extract_layers(rect: np.ndarray) -> list[dict]:
    top, left = 0, 0
    bottom, right = map(int, rect.shape)
    layers: list[dict] = []

    while top < bottom and left < right:
        color = _boundary_color(rect, top, bottom, left, right)
        layers.append({"bounds": (top, bottom, left, right), "color": int(color)})
        if np.all(rect[top:bottom, left:right] == color):
            break

        t_top = _count_top_rows(rect, top, bottom, left, right, color)
        t_bottom = _count_bottom_rows(rect, top, bottom, left, right, color)
        t_left = _count_left_cols(rect, top, bottom, left, right, color)
        t_right = _count_right_cols(rect, top, bottom, left, right, color)
        assert t_top == t_bottom == t_left == t_right, "layer thickness must be symmetric"
        thickness = int(t_top)
        assert thickness >= 1, "non-innermost layer must peel at least one cell"

        next_top = top + thickness
        next_bottom = bottom - thickness
        next_left = left + thickness
        next_right = right - thickness
        assert next_top < next_bottom and next_left < next_right, "peeling must leave an inner core"
        top, bottom, left, right = next_top, next_bottom, next_left, next_right

    assert len(layers) >= 2, "expected at least two nested layers"
    colors = [layer["color"] for layer in layers]
    assert len(colors) == len(set(colors)), "layer colors within one rectangle must be distinct"
    return layers


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    h, _w = g.shape

    components = components_mask(g != bg)
    assert components, "expected foreground objects"

    bars: list[dict] = []
    rects: list[dict] = []
    for coords in components:
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        is_filled = int(coords.shape[0]) == patch.size
        assert is_filled, "each object must fill its bounding box"

        color_values = g[coords[:, 0], coords[:, 1]]
        unique_colors = np.unique(color_values)
        if r1 == h - 1 and c0 == c1:
            assert unique_colors.size == 1, "instruction bars must be monochrome"
            bars.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "height": int(r1 - r0 + 1),
                    "left": int(c0),
                    "color": int(unique_colors[0]),
                    "coords": coords,
                }
            )
        else:
            assert r1 < h - 1, "nested rectangles must not touch the bottom edge"
            layers = _extract_layers(patch)
            rects.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "left": int(c0),
                    "patch": patch,
                    "layers": layers,
                }
            )

    assert len(bars) == len(rects) >= 2, "bars and nested rectangles must form equal nontrivial sets"
    bars.sort(key=lambda item: item["left"])
    rects.sort(key=lambda item: item["left"])

    for bar, rect in zip(bars, rects):
        outer_color = int(rect["layers"][0]["color"])
        assert bar["color"] == outer_color, "each bar color must match its rectangle outer layer"

    return bg, bars, rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle each nested rectangle's layer colors inward by the height of its corresponding bottom bar.

    Input structure:
    - The background is the most common color.
    - The upper part of the grid contains several separated solid rectangles made of concentric single-color rectangular layers.
    - The bottom edge contains the same number of one-cell-wide monochrome vertical bars; from left to right they correspond to the rectangles from left to right, and each bar has the same color as its rectangle's current outer layer.

    Rule:
    - Identify the bottom-touching width-1 columns as instruction bars and all other foreground components as nested rectangles.
    - For each rectangle, read the layer colors from outermost to innermost by repeatedly peeling the current uniform rectangular boundary.
    - If the corresponding bar has height k, rotate that outer-to-inner color sequence inward by k positions, repaint the same geometric layers with the rotated sequence, and erase every instruction bar.

    Key insight:
    - The bars are not output objects but shift controls: their heights choose a cyclic permutation of whole rectangular color layers, so geometry stays fixed while colors move through the nesting order.
    """
    g = np.asarray(grid, dtype=int)
    bg, bars, rects = _analyze_scene(g)

    out = g.copy()
    for bar in bars:
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for bar, rect in zip(bars, rects):
        layers = rect["layers"]
        colors = [int(layer["color"]) for layer in layers]
        shift = int(bar["height"]) % len(colors)
        rotated = colors[-shift:] + colors[:-shift] if shift else colors

        r0, c0, r1, c1 = rect["bbox"]
        out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        for layer, color in zip(layers, rotated):
            top, bottom, left, right = layer["bounds"]
            out_patch[top:bottom, left:right] = int(color)

    return out


PALETTE = list(range(1, 10))
GRID_MIN = 18
GRID_MAX = 28
RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (5, 11)
LAYER_COUNT_RANGE = (3, 5)
MAX_LAYER_THICKNESS = 2
BAR_HEIGHT_MAX = 8
MAX_ATTEMPTS = 500


def _pick_thicknesses(height: int, width: int, layer_count: int) -> list[int] | None:
    thicks: list[int] = []
    cur_h, cur_w = int(height), int(width)
    for i in range(layer_count - 1):
        remaining_layers = layer_count - i - 1
        min_after = 2 * (remaining_layers - 1) + 1
        max_t = min(MAX_LAYER_THICKNESS, (min(cur_h, cur_w) - min_after) // 2)
        if max_t < 1:
            return None
        thickness = random.randint(1, max_t)
        thicks.append(int(thickness))
        cur_h -= 2 * thickness
        cur_w -= 2 * thickness
    return thicks


def _build_rect_patch(height: int, width: int, colors: list[int], thicks: list[int]) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    top, left = 0, 0
    bottom, right = int(height), int(width)
    for idx, color in enumerate(colors):
        patch[top:bottom, left:right] = int(color)
        if idx < len(thicks):
            thickness = int(thicks[idx])
            top += thickness
            bottom -= thickness
            left += thickness
            right -= thickness
    return patch


def _choose_outer_colors(count: int) -> list[int]:
    if count >= 3 and random.random() < 0.6:
        outer = [random.choice(PALETTE) for _ in range(count)]
        dup_i, dup_j = random.sample(range(count), 2)
        repeated = random.choice(PALETTE)
        outer[dup_i] = repeated
        outer[dup_j] = repeated
        return outer
    return [random.choice(PALETTE) for _ in range(count)]


def _unchanged_height(layer_count: int) -> int:
    candidates = [layer_count, 2 * layer_count]
    candidates = [height for height in candidates if height <= BAR_HEIGHT_MAX]
    return int(random.choice(candidates or [layer_count]))


def _changed_height(layer_count: int) -> int:
    candidates = [height for height in range(1, BAR_HEIGHT_MAX + 1) if height % layer_count != 0]
    high = [height for height in candidates if height > layer_count]
    if high and random.random() < 0.7:
        return int(random.choice(high))
    return int(random.choice(candidates))


def _pick_bar_columns(width: int, count: int) -> list[int] | None:
    for _ in range(200):
        cols = sorted(random.sample(range(width), count))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(count - 1)):
            return cols
    return None


def _make_rect_specs(count: int) -> list[dict] | None:
    specs: list[dict] = []
    for _ in range(count):
        height = random.randint(*RECT_SIDE_RANGE)
        width = random.randint(*RECT_SIDE_RANGE)
        max_layers = min(LAYER_COUNT_RANGE[1], (min(height, width) + 1) // 2)
        if max_layers < LAYER_COUNT_RANGE[0]:
            return None
        layer_count = random.randint(LAYER_COUNT_RANGE[0], max_layers)
        thicks = _pick_thicknesses(height, width, layer_count)
        if thicks is None:
            return None
        specs.append(
            {
                "height": int(height),
                "width": int(width),
                "layer_count": int(layer_count),
                "thicks": thicks,
            }
        )

    outer_colors = _choose_outer_colors(count)
    for spec, outer_color in zip(specs, outer_colors):
        inner_choices = [color for color in PALETTE if color != outer_color]
        inner_colors = random.sample(inner_choices, spec["layer_count"] - 1)
        colors = [int(outer_color), *map(int, inner_colors)]
        patch = _build_rect_patch(spec["height"], spec["width"], colors, spec["thicks"])
        spec["colors"] = colors
        spec["patch"] = patch
    return specs


def _layout_rectangles(specs: list[dict], grid_h: int, grid_w: int, gap_to_bars: int, max_bar_height: int) -> bool:
    widths = [int(spec["width"]) for spec in specs]
    rect_gaps = [random.randint(1, 3) for _ in range(len(specs) - 1)]
    total_width = sum(widths) + sum(rect_gaps)
    if total_width > grid_w:
        return False

    start_col = random.randint(0, grid_w - total_width)
    cur_col = start_col
    for idx, spec in enumerate(specs):
        max_top = grid_h - max_bar_height - gap_to_bars - spec["height"]
        if max_top < 0:
            return False
        spec["top"] = random.randint(0, max_top)
        spec["left"] = int(cur_col)
        cur_col += spec["width"]
        if idx < len(rect_gaps):
            cur_col += rect_gaps[idx]
    return True


def _paint_scene(specs: list[dict], bar_columns: list[int], bar_heights: list[int], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        patch = spec["patch"]
        h, w = patch.shape
        grid[top : top + h, left : left + w] = patch

    for spec, col, height in zip(specs, bar_columns, bar_heights):
        grid[grid_h - height : grid_h, int(col)] = int(spec["colors"][0])
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg, bars, rects = _analyze_scene(grid)
    except AssertionError:
        return False

    if bg != 0:
        return False
    changed = 0
    unchanged = 0
    outer_colors = []
    for bar, rect in zip(bars, rects):
        layer_count = len(rect["layers"])
        if bar["height"] % layer_count == 0:
            unchanged += 1
        else:
            changed += 1
        outer_colors.append(int(rect["layers"][0]["color"]))
    if changed < 1 or unchanged < 1:
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.any(out[-1, :] != 0):
        return False
    if len(set(outer_colors)) == len(outer_colors) and len(rects) >= 3 and random.random() < 0.25:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    specs = _make_rect_specs(count)
    if specs is None:
        return None

    layer_counts = [spec["layer_count"] for spec in specs]
    unchanged_idx = random.randrange(count)
    changed_candidates = [idx for idx in range(count) if idx != unchanged_idx]
    changed_idx = random.choice(changed_candidates)
    bar_heights = [0] * count
    for idx, layer_count in enumerate(layer_counts):
        if idx == unchanged_idx:
            bar_heights[idx] = _unchanged_height(layer_count)
        elif idx == changed_idx:
            bar_heights[idx] = _changed_height(layer_count)
        else:
            if random.random() < 0.4:
                bar_heights[idx] = _unchanged_height(layer_count)
            else:
                bar_heights[idx] = _changed_height(layer_count)

    max_bar_height = max(bar_heights)
    gap_to_bars = random.randint(1, 3)
    min_h = max(GRID_MIN, max(spec["height"] for spec in specs) + max_bar_height + gap_to_bars)
    if min_h > GRID_MAX:
        return None
    grid_h = random.randint(min_h, GRID_MAX)

    widths = [spec["width"] for spec in specs]
    min_w = max(GRID_MIN, sum(widths) + (count - 1), 2 * count - 1)
    if min_w > GRID_MAX:
        return None
    grid_w = random.randint(min_w, GRID_MAX)

    if not _layout_rectangles(specs, grid_h, grid_w, gap_to_bars, max_bar_height):
        return None
    bar_columns = _pick_bar_columns(grid_w, count)
    if bar_columns is None:
        return None

    grid = _paint_scene(specs, bar_columns, bar_heights, grid_h, grid_w)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-layer bar-shift input")
