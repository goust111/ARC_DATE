from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BAR = 9
FOREGROUND_COLORS = [1, 2, 3, 4, 6, 7, 8]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) != int(color):
                continue
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) in (BG, BAR):
                continue
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            colors = set()
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                colors.add(int(grid[r, c]))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) not in (BG, BAR):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                    "colors": tuple(sorted(colors)),
                }
            )
    return out


def _find_bars(grid: np.ndarray) -> list[dict]:
    bars = _find_color_components(grid, BAR)
    for bar in bars:
        r0, c0, r1, c1 = bar["bbox"]
        assert r0 == r1, "control bars must be horizontal"
        assert bar["area"] == c1 - c0 + 1, "control bars must be contiguous segments"
    return bars


def _swap_values(values: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = values.copy()
    out[values == int(color_a)] = int(color_b)
    out[values == int(color_b)] = int(color_a)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use span-matched control bars to split, recolor, and vertically slide each paired two-color object.

    Input structure:
    - The grid has black background, several maroon one-pixel-thick horizontal control bars, and 2-3 non-black/non-maroon 4-connected objects.
    - Each foreground object uses exactly two colors and has an even-height bounding box.
    - Every control bar has exactly the same column span as one unique paired two-color object, and lies 1-3 rows directly above or below that object.

    Rule:
    - Find all maroon horizontal bars and all non-black/non-maroon 4-connected two-color objects.
    - Pair each bar with the unique object whose bounding-box column span matches the bar exactly, then erase the bar.
    - Split the paired object by the horizontal midpoint of its bounding box into upper and lower halves, and swap its two colors in both halves.
    - If the bar is above the object, slide the upper half upward until its bottom row becomes adjacent below the bar's former row; otherwise slide the lower half downward until its top row becomes adjacent above the bar's former row. The other half stays fixed, and vacated cells become background.

    Key insight:
    - The maroon bar simultaneously identifies which object to operate on, encodes whether the moving half goes upward or downward, and provides the cleared destination line after the bar is erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bars = _find_bars(g)
    objects = _find_objects(g)
    assert len(bars) == len(objects) and len(bars) >= 1

    span_to_object: dict[tuple[int, int], dict] = {}
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        height = r1 - r0 + 1
        assert height % 2 == 0, "object bbox height must be even"
        assert len(obj["colors"]) == 2, "object must be bicolor"
        span = (c0, c1)
        assert span not in span_to_object, "column spans must pair uniquely"
        span_to_object[span] = obj

    out = g.copy()
    for bar in bars:
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for bar in bars:
        br0, bc0, _br1, bc1 = bar["bbox"]
        span = (bc0, bc1)
        assert span in span_to_object, "each bar must match one object by exact span"
        obj = span_to_object[span]
        r0, _c0, r1, _c1 = obj["bbox"]
        color_a, color_b = obj["colors"]
        height = r1 - r0 + 1
        split_row = r0 + height // 2

        coords = obj["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        values = g[rows, cols]
        swapped = _swap_values(values, color_a, color_b)

        upper_mask = rows < split_row
        lower_mask = ~upper_mask
        assert upper_mask.any() and lower_mask.any()

        upper_rows = rows[upper_mask]
        upper_cols = cols[upper_mask]
        upper_vals = swapped[upper_mask]
        lower_rows = rows[lower_mask]
        lower_cols = cols[lower_mask]
        lower_vals = swapped[lower_mask]

        if br0 < r0:
            shift = (br0 + 1) - int(upper_rows.max())
            upper_rows = upper_rows + shift
        else:
            assert br0 > r1
            shift = (br0 - 1) - int(lower_rows.min())
            lower_rows = lower_rows + shift

        all_rows = np.concatenate([upper_rows, lower_rows])
        all_cols = np.concatenate([upper_cols, lower_cols])
        assert (0 <= all_rows).all() and (all_rows < g.shape[0]).all()
        assert (0 <= all_cols).all() and (all_cols < g.shape[1]).all()

        out[upper_rows, upper_cols] = upper_vals
        out[lower_rows, lower_cols] = lower_vals

    return out


_TEMPLATES = {
    2: [
        np.array(
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=int,
        ),
    ],
    3: [
        np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=int,
        ),
    ],
    4: [
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
            ],
            dtype=int,
        ),
    ],
    5: [
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1],
            ],
            dtype=int,
        ),
    ],
}


def _colorize_mask(mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    h, w = mask.shape
    mode = random.choice(["checker", "columns"])
    phase = random.randint(0, 1)
    patch = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if mode == "checker":
                choose_a = (r + c + phase) % 2 == 0
            else:
                choose_a = (c + phase) % 2 == 0
            patch[r, c] = color_a if choose_a else color_b
    used = set(int(v) for v in patch[patch != 0])
    assert used == {int(color_a), int(color_b)}
    return patch


def _expected_from_plan(shape: tuple[int, int], plans: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for plan in plans:
        patch = plan["patch"]
        h, w = patch.shape
        half_h = h // 2
        colors = sorted(set(int(v) for v in patch[patch != 0]))
        swapped = _swap_values(patch, colors[0], colors[1])
        r0 = plan["row"]
        c0 = plan["col"]
        gap = plan["gap"]

        top = swapped[:half_h]
        bottom = swapped[half_h:]
        if plan["direction"] == "up":
            top_r = r0 - gap - 1
            bottom_r = r0 + half_h
        else:
            top_r = r0
            bottom_r = r0 + h + gap - 1

        for rr in range(half_h):
            for cc in range(w):
                if top[rr, cc] != 0:
                    out[top_r + rr, c0 + cc] = int(top[rr, cc])
                if bottom[rr, cc] != 0:
                    out[bottom_r + rr, c0 + cc] = int(bottom[rr, cc])
    return out


def _sample_widths(count: int) -> list[int]:
    return random.sample([2, 3, 4, 5], count)


def _sample_directions(count: int) -> list[str]:
    if count == 2:
        dirs = ["up", "down"]
    else:
        dirs = ["up", "down", random.choice(["up", "down"])]
    random.shuffle(dirs)
    return dirs


def _make_plan(width: int, direction: str) -> dict:
    mask = np.array(random.choice(_TEMPLATES[width]), dtype=int)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    color_a, color_b = random.sample(FOREGROUND_COLORS, 2)
    patch = _colorize_mask(mask, color_a, color_b)
    return {
        "width": int(width),
        "height": int(mask.shape[0]),
        "direction": direction,
        "gap": random.randint(1, 3),
        "patch": patch,
    }


def _assign_columns(plans: list[dict], width: int) -> bool:
    order = list(range(len(plans)))
    random.shuffle(order)
    permuted = [plans[i] for i in order]

    min_total = 1 + sum(plan["width"] for plan in permuted) + (len(permuted) - 1) * 2 + 1
    if min_total > width:
        return False

    extra = width - min_total
    slots = len(permuted) + 1
    bonuses = [0] * slots
    for _ in range(extra):
        bonuses[random.randrange(slots)] += 1

    c = 1 + bonuses[0]
    for i, plan in enumerate(permuted):
        plan["col"] = c
        c += plan["width"]
        if i + 1 < len(permuted):
            c += 2 + bonuses[i + 1]
    return True


def _assign_rows(plans: list[dict], height: int) -> bool:
    for plan in plans:
        h = plan["height"]
        gap = plan["gap"]
        if plan["direction"] == "up":
            lo = gap + 1
            hi = height - h - 1
        else:
            lo = 1
            hi = height - h - gap - 1
        if lo > hi:
            return False
        plan["row"] = random.randint(lo, hi)
    return True


def _render_input(shape: tuple[int, int], plans: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for plan in plans:
        patch = plan["patch"]
        r0 = plan["row"]
        c0 = plan["col"]
        h, w = patch.shape
        for rr in range(h):
            for cc in range(w):
                value = int(patch[rr, cc])
                if value != 0:
                    grid[r0 + rr, c0 + cc] = value
        if plan["direction"] == "up":
            bar_row = r0 - plan["gap"] - 1
        else:
            bar_row = r0 + h + plan["gap"]
        plan["bar_row"] = int(bar_row)
        grid[bar_row, c0 : c0 + w] = BAR
    return grid


def _input_semantics_ok(grid: np.ndarray, expected: np.ndarray, plans: list[dict]) -> bool:
    bars = _find_bars(grid)
    objects = _find_objects(grid)
    if len(bars) != len(plans) or len(objects) != len(plans):
        return False

    spans = set()
    for obj in objects:
        if len(obj["colors"]) != 2:
            return False
        r0, c0, r1, c1 = obj["bbox"]
        if (r1 - r0 + 1) % 2 != 0:
            return False
        span = (c0, c1)
        if span in spans:
            return False
        spans.add(span)

    directions = {plan["direction"] for plan in plans}
    if len(plans) >= 2 and directions != {"up", "down"}:
        return False

    if np.array_equal(grid, expected):
        return False

    for plan in plans:
        c0 = plan["col"]
        c1 = c0 + plan["width"] - 1
        bar_row = plan["bar_row"]
        matched = [obj for obj in objects if obj["bbox"][1] == c0 and obj["bbox"][3] == c1]
        if len(matched) != 1:
            return False
        obj = matched[0]
        if plan["direction"] == "up":
            if not (bar_row < obj["bbox"][0]):
                return False
        else:
            if not (bar_row > obj["bbox"][2]):
                return False

    return np.array_equal(main(grid), expected)


def _generate_attempt() -> np.ndarray | None:
    count = random.choice([2, 3])
    widths = _sample_widths(count)
    directions = _sample_directions(count)
    plans = [_make_plan(width, direction) for width, direction in zip(widths, directions)]

    min_width = 1 + sum(plan["width"] for plan in plans) + (count - 1) * 2 + 1
    grid_w = random.randint(max(12, min_width), min(18, max(12, min_width + 5)))
    grid_h = random.randint(16, 22)

    if not _assign_columns(plans, grid_w):
        return None
    if not _assign_rows(plans, grid_h):
        return None

    inp = _render_input((grid_h, grid_w), plans)
    expected = _expected_from_plan(inp.shape, plans)
    if not _input_semantics_ok(inp, expected, plans):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
