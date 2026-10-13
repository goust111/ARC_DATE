from __future__ import annotations

import random
from collections import deque

import numpy as np


TEMP_CANVAS = 40
LAYER_COUNTS = (3, 4)
BASE_RECT_RANGE = (7, 12)
UPPER_RECT_RANGE = (6, 10)
PADDING_RANGE = (1, 3)
PLACEMENT_RETRIES = 200
MAX_GENERATION_ATTEMPTS = 300
MAX_GRID_SIDE = 30
MIN_VISIBLE_PIXELS = 8
MIN_OVERLAP_PIXELS = 8


def _background_color(grid: np.ndarray) -> int:
    values = np.unique(grid)
    if 0 in values:
        return 0
    uniq, counts = np.unique(grid, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "bbox requires a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _apply_transform(arr: np.ndarray, mode: int) -> np.ndarray:
    out = arr
    if mode >= 4:
        out = out.T
    if mode & 1:
        out = np.flipud(out)
    if mode & 2:
        out = np.fliplr(out)
    return out.copy()


def _solve_layers(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground layer"

    layers: list[dict[str, object]] = []
    for color in colors:
        visible = g == color
        assert _component_count(visible) == 1, "each color must remain 4-connected"
        bbox = _bbox(visible)
        full = _rect_mask(g.shape, bbox)
        layers.append(
            {
                "color": color,
                "visible": visible,
                "bbox": bbox,
                "full": full,
            }
        )

    n = len(layers)
    above: list[set[int]] = [set() for _ in range(n)]
    indegree = [0] * n
    for lower in range(n):
        lower_full = layers[lower]["full"]
        for upper in range(n):
            if lower == upper:
                continue
            if np.any(lower_full & layers[upper]["visible"]):
                above[lower].add(upper)

    for lower in range(n):
        for upper in above[lower]:
            assert lower not in above[upper], "occlusion relation must be acyclic"
            indegree[upper] += 1

    order: list[int] = []
    frontier = [idx for idx, deg in enumerate(indegree) if deg == 0]
    while frontier:
        assert len(frontier) == 1, "occlusion order must be uniquely determined"
        idx = frontier.pop()
        order.append(idx)
        for nxt in sorted(above[idx]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(nxt)

    assert len(order) == n, "failed to recover a full layer order"
    return bg, [layers[idx] for idx in order]


def _render_output(bg: int, ordered_layers: list[dict[str, object]]) -> np.ndarray:
    sizes = []
    for layer in ordered_layers:
        r0, c0, r1, c1 = layer["bbox"]
        sizes.append((r1 - r0 + 1, c1 - c0 + 1))

    out_h = max(h for h, _ in sizes)
    out_w = sum(w for _, w in sizes) + len(sizes) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for (h, w), layer in zip(sizes, ordered_layers):
        out[:h, col : col + w] = int(layer["color"])
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full stacked layers, sort them from back to front, and pack them left-to-right.

    In this generated family each monochrome layer is a solid rectangle. The visible
    fragment of each color stays 4-connected, and any hidden part can be restored by
    extending the visible fragment back to its full bounding rectangle.
    """
    bg, ordered_layers = _solve_layers(np.asarray(grid, dtype=int))
    return _render_output(bg, ordered_layers)


def _ground_truth_output(colors: list[int], masks: list[np.ndarray], bg: int) -> np.ndarray:
    bboxes = [_bbox(mask) for mask in masks]
    sizes = [(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in bboxes]
    out_h = max(h for h, _ in sizes)
    out_w = sum(w for _, w in sizes) + len(sizes) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for color, (h, w) in zip(colors, sizes):
        out[:h, col : col + w] = int(color)
        col += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.choice(LAYER_COUNTS)
    colors = random.sample(range(1, 10), n_layers)

    base_h = random.randint(*BASE_RECT_RANGE)
    base_w = random.randint(*BASE_RECT_RANGE)
    base_bbox = (4, 4, 4 + base_h - 1, 4 + base_w - 1)
    bboxes = [base_bbox]
    masks = [_rect_mask((TEMP_CANVAS, TEMP_CANVAS), base_bbox)]

    for _ in range(1, n_layers):
        prev_r0, prev_c0, prev_r1, prev_c1 = bboxes[-1]
        prev_h = prev_r1 - prev_r0 + 1
        prev_w = prev_c1 - prev_c0 + 1

        found = False
        for _ in range(PLACEMENT_RETRIES):
            h = random.randint(*UPPER_RECT_RANGE)
            w = random.randint(*UPPER_RECT_RANGE)
            dr_min = max(2, prev_h // 2)
            dc_min = max(2, prev_w // 2)
            dr_max = max(dr_min, prev_h - 2)
            dc_max = max(dc_min, prev_w - 2)
            dr = random.randint(dr_min, dr_max)
            dc = random.randint(dc_min, dc_max)

            bbox = (prev_r0 + dr, prev_c0 + dc, prev_r0 + dr + h - 1, prev_c0 + dc + w - 1)
            if bbox[2] >= TEMP_CANVAS - 2 or bbox[3] >= TEMP_CANVAS - 2:
                continue

            mask = _rect_mask((TEMP_CANVAS, TEMP_CANVAS), bbox)
            if np.count_nonzero(mask & masks[-1]) < MIN_OVERLAP_PIXELS:
                continue
            if any(np.any(mask & older) for older in masks[:-1]):
                continue

            bboxes.append(bbox)
            masks.append(mask)
            found = True
            break

        if not found:
            return None

    mode = random.randrange(8)
    masks = [_apply_transform(mask, mode) for mask in masks]

    grid = np.zeros((TEMP_CANVAS, TEMP_CANVAS), dtype=int)
    for color, mask in zip(colors, masks):
        grid[mask] = int(color)

    used = grid != 0
    r0, c0, r1, c1 = _bbox(used)
    pad_top = random.randint(*PADDING_RANGE)
    pad_left = random.randint(*PADDING_RANGE)
    pad_bottom = random.randint(*PADDING_RANGE)
    pad_right = random.randint(*PADDING_RANGE)
    r0 = max(0, r0 - pad_top)
    c0 = max(0, c0 - pad_left)
    r1 = min(TEMP_CANVAS - 1, r1 + pad_bottom)
    c1 = min(TEMP_CANVAS - 1, c1 + pad_right)

    grid = grid[r0 : r1 + 1, c0 : c1 + 1]
    masks = [mask[r0 : r1 + 1, c0 : c1 + 1] for mask in masks]

    if grid.shape[0] > MAX_GRID_SIDE or grid.shape[1] > MAX_GRID_SIDE:
        return None

    for color, mask in zip(colors, masks):
        visible = grid == color
        if _component_count(visible) != 1:
            return None
        if _bbox(visible) != _bbox(mask):
            return None
        if int(np.count_nonzero(visible)) < MIN_VISIBLE_PIXELS:
            return None

    try:
        bg, ordered_layers = _solve_layers(grid)
    except AssertionError:
        return None

    if [int(layer["color"]) for layer in ordered_layers] != colors:
        return None

    expected = _ground_truth_output(colors, masks, bg=0)
    if not np.array_equal(_render_output(bg, ordered_layers), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
