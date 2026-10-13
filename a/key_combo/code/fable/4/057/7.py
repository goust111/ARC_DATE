from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int]) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) not in exclude)


def _group_consecutive(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _find_occluder(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int, int]] = []

    for color in _colors_present(grid, exclude={bg}):
        full_cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        for c0, c1 in _group_consecutive(full_cols):
            allowed = np.zeros_like(grid, dtype=bool)
            allowed[:, c0 : c1 + 1] = True
            color_mask = grid == color
            if np.any(color_mask & ~allowed):
                continue
            candidates.append((color, c0, c1))

    assert len(candidates) == 1, "expected a unique full-height vertical occluder bar"
    return candidates[0]


def _is_connected(coords: set[Coord]) -> bool:
    if not coords:
        return False
    stack = [next(iter(coords))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _dedupe_candidates(candidates: list[set[Coord]]) -> list[set[Coord]]:
    unique: dict[frozenset[Coord], set[Coord]] = {}
    for coords in candidates:
        unique.setdefault(frozenset(coords), coords)
    return list(unique.values())


def _vertical_candidates(
    observed: set[Coord], height: int, width: int, bar_cols: set[int], require_hidden: bool
) -> list[set[Coord]]:
    results: list[set[Coord]] = []
    for axis_sum in range(0, 2 * width - 1):
        added: set[Coord] = set()
        ok = True
        for r, c in observed:
            cc = axis_sum - c
            if not 0 <= cc < width:
                ok = False
                break
            mate = (r, int(cc))
            if mate not in observed:
                if int(cc) not in bar_cols:
                    ok = False
                    break
                added.add(mate)
        if not ok or (require_hidden and not added):
            continue
        full = observed | added
        if _is_connected(full):
            results.append(full)
    return _dedupe_candidates(results)


def _rot180_candidates(
    observed: set[Coord], height: int, width: int, bar_cols: set[int], require_hidden: bool
) -> list[set[Coord]]:
    results: list[set[Coord]] = []
    for row_sum in range(0, 2 * height - 1):
        for col_sum in range(0, 2 * width - 1):
            added: set[Coord] = set()
            ok = True
            for r, c in observed:
                rr = row_sum - r
                cc = col_sum - c
                if not (0 <= rr < height and 0 <= cc < width):
                    ok = False
                    break
                mate = (int(rr), int(cc))
                if mate not in observed:
                    if int(cc) not in bar_cols:
                        ok = False
                        break
                    added.add(mate)
            if not ok or (require_hidden and not added):
                continue
            full = observed | added
            if _is_connected(full):
                results.append(full)
    return _dedupe_candidates(results)


def _reconstruct_object(observed: set[Coord], shape: tuple[int, int], bar_cols: set[int]) -> set[Coord]:
    height, width = map(int, shape)
    candidates = _vertical_candidates(observed, height, width, bar_cols, require_hidden=True)
    candidates.extend(_rot180_candidates(observed, height, width, bar_cols, require_hidden=True))
    candidates = _dedupe_candidates(candidates)
    if not candidates:
        candidates = _vertical_candidates(observed, height, width, bar_cols, require_hidden=False)
        candidates.extend(_rot180_candidates(observed, height, width, bar_cols, require_hidden=False))
        candidates = _dedupe_candidates(candidates)
    assert len(candidates) == 1, "expected a unique symmetry completion for each object"
    return candidates[0]


def _crop_coords(coords: set[Coord], color: int, bg: int) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each bar-occluded object by its own symmetry type, then crop the uniquely largest inferred shape.

    Input structure:
    - One dominant background surrounds several separated monochrome objects of distinct colors.
    - A unique full-height solid vertical bar overwrites some object cells.
    - Each underlying object is either vertically mirror-symmetric or 180-degree rotationally symmetric.

    Rule:
    - Identify the occluder bar and ignore its color when grouping object pixels.
    - For each remaining object color, infer the unique symmetry completion whose missing counterpart cells lie only inside the bar.
    - Compare the completed objects by inferred area, keep the unique largest one, and return its tight crop with background elsewhere.

    Key insight:
    - The visible fragments must be completed object-by-object: the correct winner is determined only after testing both symmetry families and restoring the bar-hidden cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    bg = _get_bg(g)
    bar_color, bar_c0, bar_c1 = _find_occluder(g, bg)
    bar_cols = set(range(bar_c0, bar_c1 + 1))

    colors = _colors_present(g, exclude={bg, bar_color})
    assert colors, "expected at least one foreground object"

    reconstructions: list[tuple[int, set[Coord]]] = []
    for color in colors:
        observed = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
        assert observed, "each object color must have visible cells"
        full = _reconstruct_object(observed, (height, width), bar_cols)
        reconstructions.append((color, full))

    areas = [len(coords) for _color, coords in reconstructions]
    max_area = max(areas)
    largest = [(color, coords) for color, coords in reconstructions if len(coords) == max_area]
    assert len(largest) == 1, "expected a unique largest completed object"

    color, coords = largest[0]
    return _crop_coords(coords, color, bg)


# --------------- generator ---------------

BG_COLOR = 0
PALETTE = list(range(1, 10))
GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (20, 29)
BAR_WIDTH_RANGE = (2, 4)
OBJECT_COUNT_RANGE = (3, 5)
HEIGHT_RANGE = (4, 7)
HALF_WIDTH_MAX = 8
MAX_ATTEMPTS = 400


def _interval_mask(height: int, width: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for c, (r0, r1) in enumerate(intervals):
        mask[r0 : r1 + 1, c] = True
    return mask


def _random_intervals(height: int, width: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = [(0, height - 1)]
    prev_r0, prev_r1 = 0, height - 1
    for _ in range(width - 1):
        overlap_lo = max(0, prev_r0 - 1)
        overlap_hi = min(height - 1, prev_r1 + 1)
        anchor = random.randint(overlap_lo, overlap_hi)
        up_room = random.randint(0, anchor)
        down_room = random.randint(0, height - 1 - anchor)
        r0 = anchor - up_room
        r1 = anchor + down_room
        intervals.append((r0, r1))
        prev_r0, prev_r1 = r0, r1
    intervals.reverse()
    intervals[-1] = (0, height - 1)
    return intervals


def _valid_half_mask(mask: np.ndarray, hidden_width: int) -> bool:
    height, width = mask.shape
    if width <= hidden_width + 1:
        return False
    if not np.all(mask[:, -1]):
        return False
    if np.any(mask.sum(axis=0) == 0):
        return False
    if np.array_equal(mask, mask[::-1, :]):
        return False
    for c in range(width - hidden_width):
        col = mask[:, c]
        if not np.array_equal(col, col[::-1]):
            return True
    return False


def _make_half_mask(height: int, width: int, hidden_width: int) -> np.ndarray | None:
    for _ in range(80):
        intervals = _random_intervals(height, width)
        mask = _interval_mask(height, width, intervals)
        if _valid_half_mask(mask, hidden_width):
            return mask
    return None


def _build_full_mask(half_mask: np.ndarray, symmetry: str) -> np.ndarray:
    if symmetry == "vertical":
        right = half_mask[:, ::-1]
    else:
        right = half_mask[::-1, ::-1]
    return np.concatenate([half_mask, right], axis=1)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.argwhere(mask).T
    grid[top + rows, left + cols] = int(color)


def _crop_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    rows, cols = np.argwhere(mask).T
    r0, c0 = rows.min(), cols.min()
    r1, c1 = rows.max(), cols.max()
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    out[rows - r0, cols - c0] = int(color)
    return out


def _row_layout(height: int, object_heights: list[int]) -> list[int] | None:
    total_object = sum(object_heights)
    min_total = total_object + (len(object_heights) - 1)
    if min_total > height:
        return None
    slack = height - min_total
    gaps = [0] * (len(object_heights) + 1)
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1
    starts: list[int] = []
    cursor = gaps[0]
    for i, obj_h in enumerate(object_heights):
        starts.append(cursor)
        cursor += obj_h + 1 + gaps[i + 1]
    return starts


def _gen_object_specs(height: int, width: int, bar_c0: int, bar_w: int) -> list[dict] | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    types = ["vertical", "rot180"]
    while len(types) < count:
        types.append(random.choice(["vertical", "rot180"]))
    random.shuffle(types)

    colors = random.sample(PALETTE, count + 1)
    bar_color = colors[0]
    object_colors = colors[1:]

    max_half_width = min(HALF_WIDTH_MAX, bar_c0, width - bar_c0)
    if max_half_width < bar_w + 2:
        return None

    heights = [random.randint(*HEIGHT_RANGE) for _ in range(count)]
    starts = _row_layout(height, heights)
    if starts is None:
        return None

    specs: list[dict] = []
    for idx in range(count):
        obj_h = heights[idx]
        half_w = random.randint(bar_w + 2, max_half_width)
        half_mask = _make_half_mask(obj_h, half_w, bar_w)
        if half_mask is None:
            return None
        full_mask = _build_full_mask(half_mask, types[idx])
        specs.append(
            {
                "color": object_colors[idx],
                "type": types[idx],
                "top": starts[idx],
                "left": bar_c0 - half_w,
                "mask": full_mask,
                "area": int(full_mask.sum()),
            }
        )

    if len({spec["area"] for spec in specs}) != len(specs):
        return None
    if sum(spec["area"] for spec in specs) + height * bar_w >= height * width:
        return None

    largest_area = max(spec["area"] for spec in specs)
    if sum(spec["area"] == largest_area for spec in specs) != 1:
        return None

    return [{"bar_color": bar_color}, *specs]


def _render_scene(height: int, width: int, bar_c0: int, bar_w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    scene = np.full((height, width), BG_COLOR, dtype=int)
    for spec in specs[1:]:
        _paint_mask(scene, spec["mask"], spec["top"], spec["left"], spec["color"])

    inp = scene.copy()
    inp[:, bar_c0 : bar_c0 + bar_w] = int(specs[0]["bar_color"])
    return scene, inp


def _expected_output(specs: list[dict]) -> np.ndarray:
    largest = max(specs[1:], key=lambda spec: spec["area"])
    return _crop_mask(largest["mask"], largest["color"], BG_COLOR)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bar_w = random.randint(*BAR_WIDTH_RANGE)
    bar_c0 = random.randint(6, width - 7)

    specs = _gen_object_specs(height, width, bar_c0, bar_w)
    if specs is None:
        return None

    scene, inp = _render_scene(height, width, bar_c0, bar_w, specs)
    expected = _expected_output(specs)

    bg_pixels = int(np.count_nonzero(inp == BG_COLOR))
    if bg_pixels <= max(height * bar_w, max(spec["area"] for spec in specs[1:])):
        return None

    if np.count_nonzero(scene[:, bar_c0 : bar_c0 + bar_w] != BG_COLOR) == 0:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
