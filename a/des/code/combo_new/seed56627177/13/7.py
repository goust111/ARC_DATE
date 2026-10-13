from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

# --------------- shared helpers ---------------


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).reshape(-1)))
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = set() if exclude is None else {int(v) for v in exclude}
    return [int(v) for v in np.unique(grid) if int(v) not in excluded]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in steps:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        column = grid[:, c]
        vals = np.unique(column)
        if vals.size != 1:
            continue
        color = int(vals[0])
        if color == BLACK:
            continue
        if not np.any(grid[:, :c] != BLACK):
            continue
        if not np.all(grid[:, c + 1 :] == BLACK):
            continue
        candidates.append((int(c), color))
    assert len(candidates) == 1, "expected one full-height separator column"
    return candidates[0]


def _reconstruct_layers(pattern: np.ndarray) -> list[dict[str, int]]:
    layers: list[dict[str, int]] = []
    for color in colors_present(pattern, exclude={BLACK}):
        coords = np.argwhere(pattern == int(color))
        assert coords.size > 0
        r0, c0, r1, c1 = bbox_coords(coords)
        edge_checks = (
            np.any(pattern[r0, c0 : c1 + 1] == int(color)),
            np.any(pattern[r1, c0 : c1 + 1] == int(color)),
            np.any(pattern[r0 : r1 + 1, c0] == int(color)),
            np.any(pattern[r0 : r1 + 1, c1] == int(color)),
        )
        assert all(edge_checks), "each visible layer must expose all four rectangle sides"
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        layers.append(
            {
                "color": int(color),
                "h": h,
                "w": w,
                "area": int(h * w),
                "visible": int(coords.shape[0]),
            }
        )

    areas = [layer["area"] for layer in layers]
    assert len(areas) == len(set(areas)), "layer areas must uniquely determine depth order"
    layers.sort(key=lambda layer: (-layer["area"], -layer["visible"], -layer["h"], -layer["w"]))
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose each overlapped multi-color rectangle stack and repack its monochrome layers into the blank right half.

    Input structure:
    - Black background with one full-height separator column between a populated left half and a blank right half.
    - Each left-half object is an isolated multi-color pattern made from 2 or 3 axis-aligned solid rectangles painted on top of one another.
    - Every rectangle layer has its own color, and lower layers remain partially visible around the occluding upper layers.

    Rule:
    - Find each left-half pattern and, within it, reconstruct one solid rectangle per visible color by taking the full bounding box of that color's exposed pixels.
    - Order those reconstructed rectangles from largest area to smallest area, which corresponds to bottom layer to top layer.
    - Keep the left half unchanged, clear the right half to black, and place each pattern's rectangles there from left to right with one black column between them.
    - Stack pattern groups in the right half from top to bottom following the source patterns' top-to-bottom order, inserting one black row between groups.

    Key insight:
    - Occlusion hides interior pixels but not the extreme rows and columns of each rectangle, so a color's visible bounding box recovers the full layer, and area reveals the stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    sep_col, _sep_color = _find_separator(g)

    left = g[:, :sep_col]
    pattern_components = components_mask(left != BLACK, connectivity=8)
    assert pattern_components, "expected at least one source pattern"

    patterns: list[tuple[int, int, list[dict[str, int]]]] = []
    for coords in pattern_components:
        r0, c0, r1, c1 = bbox_coords(coords)
        pattern = left[r0 : r1 + 1, c0 : c1 + 1]
        layers = _reconstruct_layers(pattern)
        patterns.append((int(r0), int(c0), layers))

    patterns.sort(key=lambda item: (item[0], item[1]))

    out = g.copy()
    out[:, sep_col + 1 :] = BLACK

    row = 0
    right_w = int(w - sep_col - 1)
    for idx, (_src_r, _src_c, layers) in enumerate(patterns):
        group_h = max(layer["h"] for layer in layers)
        group_w = sum(layer["w"] for layer in layers) + len(layers) - 1
        assert row + group_h <= h, "output groups must fit vertically"
        assert group_w <= right_w, "output groups must fit in the blank half"

        col = sep_col + 1
        for layer in layers:
            rr1 = int(row + layer["h"])
            cc1 = int(col + layer["w"])
            out[row:rr1, col:cc1] = int(layer["color"])
            col = int(cc1 + 1)

        row += group_h
        if idx + 1 < len(patterns):
            row += 1

    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
PATTERN_COUNT_RANGE = (1, 3)
LAYER_COUNT_OPTIONS = (2, 3)
TOP_SIZE_BY_LAYER_COUNT = {2: (2, 4), 3: (2, 3)}
MAX_GROUP_WIDTH = 16
MAX_OUTER_SIZE = 9
MAX_EXTRA_HEIGHT = 3
MAX_EXTRA_WIDTH = 3


def _scatter_units(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _axis_expansion() -> tuple[int, int]:
    total = random.choice((2, 2, 3))
    first = random.randint(1, total - 1)
    second = total - first
    return int(first), int(second)


def _build_rectangles(layer_count: int) -> list[tuple[int, int, int, int]] | None:
    top_min, top_max = TOP_SIZE_BY_LAYER_COUNT[layer_count]
    top_h = random.randint(top_min, top_max)
    top_w = random.randint(top_min, top_max)
    rects = [(0, 0, top_h - 1, top_w - 1)]  # inner to outer

    for _ in range(layer_count - 1):
        r0, c0, r1, c1 = rects[-1]
        add_top, add_bottom = _axis_expansion()
        add_left, add_right = _axis_expansion()
        rects.append((r0 - add_top, c0 - add_left, r1 + add_bottom, c1 + add_right))

    min_r = min(r0 for r0, _c0, _r1, _c1 in rects)
    min_c = min(c0 for _r0, c0, _r1, _c1 in rects)
    normalized = [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in rects]
    normalized.reverse()  # bottom to top

    widths = [c1 - c0 + 1 for _r0, c0, _r1, c1 in normalized]
    heights = [r1 - r0 + 1 for r0, _c0, r1, _c1 in normalized]
    group_w = sum(widths) + len(widths) - 1
    if group_w > MAX_GROUP_WIDTH:
        return None
    if max(widths) > MAX_OUTER_SIZE or max(heights) > MAX_OUTER_SIZE:
        return None
    return normalized


def _make_pattern_spec() -> dict[str, object] | None:
    layer_count = random.choice(LAYER_COUNT_OPTIONS)
    rects = _build_rectangles(layer_count)
    if rects is None:
        return None

    colors = random.sample(PALETTE, layer_count)
    layers: list[dict[str, int]] = []
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        layers.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
            }
        )

    return {
        "layers": layers,
        "h": layers[0]["h"],
        "w": layers[0]["w"],
        "group_w": sum(layer["w"] for layer in layers) + len(layers) - 1,
    }


def _random_height(pattern_heights: list[int]) -> int | None:
    used = sum(pattern_heights)
    min_gaps = max(0, len(pattern_heights) - 1)
    base = used + min_gaps
    if base > 30:
        return None
    extra = random.randint(0, min(MAX_EXTRA_HEIGHT, 30 - base))
    return int(base + extra)


def _random_width(pattern_widths: list[int], group_widths: list[int]) -> tuple[int, int] | None:
    left_min = max(pattern_widths) + 1
    right_min = max(group_widths)
    base = left_min + 1 + right_min
    if base > 30:
        return None

    left_extra_limit = min(MAX_EXTRA_WIDTH, 30 - base)
    left_w = left_min + random.randint(0, left_extra_limit)
    right_extra_limit = min(MAX_EXTRA_WIDTH, 30 - (left_w + 1 + right_min))
    right_w = right_min + random.randint(0, right_extra_limit)
    return int(left_w), int(right_w)


def _place_pattern_layers(grid: np.ndarray, top: int, left: int, layers: list[dict[str, int]]) -> None:
    for layer in layers:
        r0 = top + layer["r0"]
        r1 = top + layer["r1"] + 1
        c0 = left + layer["c0"]
        c1 = left + layer["c1"] + 1
        grid[r0:r1, c0:c1] = int(layer["color"])


def _gen_attempt() -> np.ndarray | None:
    pattern_count = random.randint(*PATTERN_COUNT_RANGE)
    specs: list[dict[str, object]] = []
    for _ in range(pattern_count):
        spec = _make_pattern_spec()
        if spec is None:
            return None
        specs.append(spec)

    pattern_heights = [int(spec["h"]) for spec in specs]
    pattern_widths = [int(spec["w"]) for spec in specs]
    group_widths = [int(spec["group_w"]) for spec in specs]

    height = _random_height(pattern_heights)
    widths = _random_width(pattern_widths, group_widths)
    if height is None or widths is None:
        return None
    left_w, right_w = widths

    total_pattern_h = sum(pattern_heights)
    free_rows = height - total_pattern_h
    between_count = pattern_count - 1
    if free_rows < between_count:
        return None

    extras = _scatter_units(free_rows - between_count, pattern_count + 1)
    gaps = [extras[0]]
    for i in range(between_count):
        gaps.append(1 + extras[i + 1])
    gaps.append(extras[-1])

    grid = np.full((height, left_w + 1 + right_w), BLACK, dtype=int)
    grid[:, left_w] = GREY

    row = gaps[0]
    for idx, spec in enumerate(specs):
        spec_w = int(spec["w"])
        max_col = left_w - spec_w - 1
        if max_col < 0:
            return None
        col = random.randint(0, max_col)
        _place_pattern_layers(grid, row, col, spec["layers"])  # type: ignore[arg-type]
        row += int(spec["h"])
        if idx + 1 < len(specs):
            row += gaps[idx + 1]

    left_components = components_mask(grid[:, :left_w] != BLACK, connectivity=8)
    if len(left_components) != pattern_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.array_equal(out[:, : left_w + 1], grid[:, : left_w + 1]):
        return None
    if np.any(grid[:, left_w + 1 :] != BLACK):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapped-rectangle puzzle")
