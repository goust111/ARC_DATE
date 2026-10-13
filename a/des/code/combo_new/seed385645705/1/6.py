from __future__ import annotations

from collections import deque
import random

import numpy as np


def _background_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _legend_pairs(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    assert h >= 2, "expected a two-row legend at the bottom"
    pairs: list[tuple[int, int]] = []
    for c in range(w):
        upper = int(g[h - 2, c])
        lower = int(g[h - 1, c])
        if upper == bg and lower == bg:
            continue
        assert upper != bg and lower != bg and upper != lower, "invalid legend pair"
        pairs.append((upper, lower))
    assert pairs, "expected at least one legend pair"
    return pairs


def _priority_order(pairs: list[tuple[int, int]]) -> list[int]:
    colors: list[int] = []
    for upper, lower in pairs:
        for color in (upper, lower):
            if color not in colors:
                colors.append(int(color))

    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for upper, lower in pairs:
        upper = int(upper)
        lower = int(lower)
        if lower not in above[upper]:
            above[upper].add(lower)
            indegree[lower] += 1

    ordered: list[int] = []
    remaining = set(colors)
    while remaining:
        ready = [color for color in colors if color in remaining and indegree[color] == 0]
        assert len(ready) == 1, "legend must define one priority chain"
        color = ready[0]
        ordered.append(color)
        remaining.remove(color)
        for nxt in above[color]:
            indegree[nxt] -= 1
    return ordered


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _stripe_mask(height: int, width: int, orientation: str, start: int, stop: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if orientation == "h":
        mask[int(start) : int(stop) + 1, :] = True
    else:
        mask[:, int(start) : int(stop) + 1] = True
    return mask


def _candidate_stripes(visible: np.ndarray, higher: np.ndarray) -> list[np.ndarray]:
    height, width = visible.shape
    candidates: list[np.ndarray] = []
    seen: set[bytes] = set()

    for r0 in range(height):
        for r1 in range(r0, height):
            mask = _stripe_mask(height, width, "h", r0, r1)
            if np.array_equal(mask & ~higher, visible):
                key = mask.tobytes()
                if key not in seen:
                    candidates.append(mask)
                    seen.add(key)

    for c0 in range(width):
        for c1 in range(c0, width):
            mask = _stripe_mask(height, width, "v", c0, c1)
            if np.array_equal(mask & ~higher, visible):
                key = mask.tobytes()
                if key not in seen:
                    candidates.append(mask)
                    seen.add(key)

    return candidates


def _reconstruct_layers(local: np.ndarray, bg: int, order_low_to_high: list[int]) -> dict[int, np.ndarray]:
    height, width = local.shape
    restored: dict[int, np.ndarray] = {}
    higher = np.zeros((height, width), dtype=bool)

    for color in reversed(order_low_to_high):
        visible = local == int(color)
        assert bool(visible.any()), "each object layer must remain partly visible"
        candidates = _candidate_stripes(visible, higher)
        assert len(candidates) == 1, "layer stripe must be uniquely recoverable"
        restored[int(color)] = candidates[0]
        higher |= candidates[0]

    return restored


def _paint_by_area(shape: tuple[int, int], bg: int, layers: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    entries = [(int(mask.sum()), color, mask) for color, mask in layers.items()]
    assert len({area for area, _color, _mask in entries}) == len(entries), "layer areas must be distinct"
    for _area, color, mask in sorted(entries, key=lambda item: -item[0]):
        out[mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore layered monochrome stripes inside each multicolor object and redraw them by area.

    Input structure:
    - The dominant color is the background.
    - Above the bottom legend, several separated multicolor rectangular objects contain two or three non-background colors.
    - Each object's hidden full color layers are horizontal or vertical rectangular strips spanning the whole object width or height.
    - The last two rows form a legend: each non-background column is a vertical pair whose upper color is hidden by the lower color.

    Rule:
    - Read the legend pairs to obtain the color priority order, with lower legend pixels above upper legend pixels.
    - For each object, process its colors from high to low priority and recover the unique full horizontal or vertical strip whose uncovered cells match the visible pixels.
    - Compute every restored strip area, then repaint the object rectangle from largest area to smallest area so smaller strips appear on top.
    - Keep the objects and the bottom legend in their original positions.

    Key insight:
    - The legend explains the input occlusion order, but the output order is not the legend order; it is determined only after restoring the full strip areas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 4 and width >= 2, "grid too small for objects and legend"

    bg = _background_color(g)
    pairs = _legend_pairs(g, bg)
    global_order = _priority_order(pairs)
    priority_index = {color: i for i, color in enumerate(global_order)}

    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[: height - 2, :] = g[: height - 2, :] != int(bg)

    out = g.copy()
    for pts in _components(object_mask):
        r0, c0 = np.min(pts, axis=0)
        r1, c1 = np.max(pts, axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        local = g[r0 : r1 + 1, c0 : c1 + 1]

        colors = [int(c) for c in np.unique(local) if int(c) != int(bg)]
        assert 2 <= len(colors) <= 3, "objects must contain two or three colors"
        assert all(color in priority_index for color in colors), "object colors must appear in the legend"

        object_order = sorted(colors, key=lambda color: priority_index[color])
        layers = _reconstruct_layers(local, bg, object_order)
        out[r0 : r1 + 1, c0 : c1 + 1] = _paint_by_area(local.shape, bg, layers)

    return out


# --------------- generator ---------------

GENERATION_ATTEMPTS = 500
COLOR_COUNT = 10
OBJECT_COUNT_RANGE = (2, 4)
COLOR_CHAIN_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (16, 28)
GRID_WIDTH_RANGE = (16, 30)
OBJECT_HEIGHT_RANGE = (4, 8)
OBJECT_WIDTH_RANGE = (4, 8)
OBJECT_PADDING = 1


def _sample_stripe(height: int, width: int, orientation: str) -> np.ndarray:
    if orientation == "h":
        thickness = random.randint(1, height - 1)
        start = random.randint(0, height - thickness)
    else:
        thickness = random.randint(1, width - 1)
        start = random.randint(0, width - thickness)
    return _stripe_mask(height, width, orientation, start, start + thickness - 1)


def _sample_two_cross_stripes(
    height: int,
    width: int,
    mid_orientation: str,
    want_high_larger: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    high_orientation = "v" if mid_orientation == "h" else "h"
    for _ in range(100):
        mid = _sample_stripe(height, width, mid_orientation)
        high = _sample_stripe(height, width, high_orientation)
        mid_area = int(mid.sum())
        high_area = int(high.sum())
        if mid_area == high_area:
            continue
        if (high_area > mid_area) == bool(want_high_larger):
            return mid, high
    return None


def _compose_priority(
    shape: tuple[int, int],
    bg: int,
    layers: dict[int, np.ndarray],
    order_low_to_high: list[int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in order_low_to_high:
        if int(color) in layers:
            out[layers[int(color)]] = int(color)
    return out


def _make_object(
    color_order: list[int],
    layer_count: int,
    force_changed: bool,
) -> tuple[np.ndarray, np.ndarray, set[int], bool] | None:
    local_bg = -1
    height = random.randint(*OBJECT_HEIGHT_RANGE)
    width = random.randint(*OBJECT_WIDTH_RANGE)

    colors = random.sample(color_order, int(layer_count))
    colors.sort(key=lambda color: color_order.index(color))
    base = int(colors[0])

    layers: dict[int, np.ndarray] = {base: np.ones((height, width), dtype=bool)}
    if layer_count == 2:
        layers[int(colors[1])] = _sample_stripe(height, width, random.choice(["h", "v"]))
    else:
        mid_orientation = random.choice(["h", "v"])
        want_high_larger = True if force_changed else random.choice([False, True])
        pair = _sample_two_cross_stripes(height, width, mid_orientation, want_high_larger)
        if pair is None:
            return None
        layers[int(colors[1])], layers[int(colors[2])] = pair

    areas = [int(mask.sum()) for mask in layers.values()]
    if len(set(areas)) != len(areas):
        return None

    local_in = _compose_priority((height, width), local_bg, layers, color_order)
    local_out = _paint_by_area((height, width), local_bg, layers)
    changed = bool(np.any(local_in != local_out))

    try:
        local_order = [color for color in color_order if color in colors]
        recovered = _reconstruct_layers(local_in, local_bg, local_order)
        if not np.array_equal(_paint_by_area(local_in.shape, local_bg, recovered), local_out):
            return None
    except Exception:
        return None

    return local_in, local_out, set(map(int, colors)), changed


def _legend_width(pair_count: int, gap: int) -> int:
    return 1 + (pair_count - 1) * (1 + gap)


def _draw_legend(grid: np.ndarray, bg: int, order_low_to_high: list[int]) -> None:
    pair_count = len(order_low_to_high) - 1
    gap = random.choice([0, 1])
    needed = _legend_width(pair_count, gap)
    start = random.randint(0, grid.shape[1] - needed)
    grid[-2:, :] = int(bg)
    for i in range(pair_count):
        col = start + i * (1 + gap)
        grid[-2, col] = int(order_low_to_high[i])
        grid[-1, col] = int(order_low_to_high[i + 1])


def _place_objects(
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    objects: list[tuple[np.ndarray, np.ndarray]],
) -> bool:
    height, width = grid_in.shape
    occupied = np.zeros((height, width), dtype=bool)
    occupied[height - 3 :, :] = True

    for local_in, local_out in random.sample(objects, len(objects)):
        oh, ow = local_in.shape
        choices: list[tuple[int, int]] = []
        for r in range(0, height - 3 - oh + 1):
            for c in range(0, width - ow + 1):
                r0 = max(0, r - OBJECT_PADDING)
                c0 = max(0, c - OBJECT_PADDING)
                r1 = min(height, r + oh + OBJECT_PADDING)
                c1 = min(width, c + ow + OBJECT_PADDING)
                if not bool(occupied[r0:r1, c0:c1].any()):
                    choices.append((r, c))
        if not choices:
            return False
        r, c = random.choice(choices)
        grid_in[r : r + oh, c : c + ow] = local_in
        grid_out[r : r + oh, c : c + ow] = local_out
        r0 = max(0, r - OBJECT_PADDING)
        c0 = max(0, c - OBJECT_PADDING)
        r1 = min(height, r + oh + OBJECT_PADDING)
        c1 = min(width, c + ow + OBJECT_PADDING)
        occupied[r0:r1, c0:c1] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(COLOR_COUNT)
    available = [color for color in range(COLOR_COUNT) if color != bg]

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    layer_counts = [2, 3]
    while len(layer_counts) < object_count:
        layer_counts.append(random.choice([2, 3]))
    random.shuffle(layer_counts)

    max_chain = min(COLOR_CHAIN_RANGE[1], sum(layer_counts), len(available))
    chain_len = random.randint(COLOR_CHAIN_RANGE[0], max_chain)
    order_low_to_high = random.sample(available, chain_len)

    objects: list[tuple[np.ndarray, np.ndarray]] = []
    used_colors: set[int] = set()
    any_changed = False
    forced_change_used = False
    for count in layer_counts:
        force_changed = count == 3 and not forced_change_used
        made = _make_object(order_low_to_high, count, force_changed)
        if made is None:
            return None
        local_in, local_out, colors, changed = made
        objects.append((local_in, local_out))
        used_colors |= colors
        any_changed = any_changed or changed
        if force_changed:
            forced_change_used = True

    if used_colors != set(order_low_to_high) or not any_changed:
        return None

    height = random.randint(*GRID_HEIGHT_RANGE)
    min_width = _legend_width(len(order_low_to_high) - 1, 1)
    width = random.randint(max(GRID_WIDTH_RANGE[0], min_width), GRID_WIDTH_RANGE[1])
    grid_in = np.full((height, width), int(bg), dtype=int)
    grid_out = np.full((height, width), int(bg), dtype=int)

    recolored_objects: list[tuple[np.ndarray, np.ndarray]] = []
    for local_in, local_out in objects:
        li = np.where(local_in == -1, int(bg), local_in)
        lo = np.where(local_out == -1, int(bg), local_out)
        recolored_objects.append((li, lo))

    if not _place_objects(grid_in, grid_out, recolored_objects):
        return None

    _draw_legend(grid_in, bg, order_low_to_high)
    grid_out[-2:, :] = grid_in[-2:, :]

    vals, counts = np.unique(grid_in, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != int(bg) or int(np.sum(counts == counts.max())) != 1:
        return None

    try:
        actual = main(grid_in)
    except Exception:
        return None
    if not np.array_equal(actual, grid_out):
        return None
    return grid_in


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-strip input")
