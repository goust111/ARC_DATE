from __future__ import annotations

import random

import numpy as np


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _color_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    current = int(values[0])
    for i in range(1, len(values) + 1):
        if i == len(values) or int(values[i]) != current:
            runs.append((current, i - start))
            if i < len(values):
                start = i
                current = int(values[i])
    return runs


def _parse_bar(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int, int, int, int, int]:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if len(rows) == 1:
        row = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert c1 - c0 + 1 == len(coords), "horizontal bar must be gapless"
        line = grid[row, c0 : c1 + 1]
        assert np.all(line != 0), "horizontal bar must occupy every cell in its span"
        center_r = row
        center_c = c0 + len(line) // 2
    elif len(cols) == 1:
        col = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        assert r1 - r0 + 1 == len(coords), "vertical bar must be gapless"
        line = grid[r0 : r1 + 1, col]
        assert np.all(line != 0), "vertical bar must occupy every cell in its span"
        center_r = r0 + len(line) // 2
        center_c = col
    else:
        raise AssertionError("each object must be a 1-cell-thick bar")

    runs = _color_runs(np.asarray(line, dtype=int))
    assert len(runs) == 2, "each bar must contain exactly two color runs"
    (color1, len1), (color2, len2) = runs
    assert color1 != color2 and min(len1, len2) >= 1

    if len1 <= len2:
        inner_color, inner_len = int(color1), int(len1)
        outer_color, outer_len = int(color2), int(len2)
    else:
        inner_color, inner_len = int(color2), int(len2)
        outer_color, outer_len = int(color1), int(len1)

    return center_r, center_c, inner_color, inner_len, outer_color, outer_len


def _paint_square(out: np.ndarray, center_r: int, center_c: int, half_len: int, color: int) -> None:
    radius = int(half_len) - 1
    r0 = max(0, center_r - radius)
    r1 = min(out.shape[0], center_r + radius + 1)
    c0 = max(0, center_c - radius)
    c1 = min(out.shape[1], center_c + radius + 1)
    out[r0:r1, c0:c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each two-color bar into its two monochrome lengths, then rebuild those
    lengths as nested filled squares centered on the original bar midpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    for coords in _components4(g != 0):
        center_r, center_c, inner_color, inner_len, outer_color, outer_len = _parse_bar(g, coords)
        _paint_square(out, center_r, center_c, outer_len, outer_color)
        _paint_square(out, center_r, center_c, inner_len, inner_color)
    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (16, 30)
GRID_WIDTH_RANGE = (16, 30)
OBJECT_COUNT_RANGE = (1, 4)
LONG_LEN_RANGE = (2, 6)
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 400


def _sample_bar_specs() -> list[dict[str, int | str]]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, 2 * count)
    specs: list[dict[str, int | str]] = []

    for i in range(count):
        outer_len = random.randint(*LONG_LEN_RANGE)
        inner_choices = [n for n in range(1, outer_len) if (n + outer_len) % 2 == 1]
        inner_len = random.choice(inner_choices)
        if random.random() < 0.5:
            first_len, second_len = inner_len, outer_len
            first_color, second_color = colors[2 * i], colors[2 * i + 1]
        else:
            first_len, second_len = outer_len, inner_len
            first_color, second_color = colors[2 * i], colors[2 * i + 1]

        specs.append(
            {
                "orientation": random.choice(("h", "v")),
                "inner_len": int(inner_len),
                "outer_len": int(outer_len),
                "first_len": int(first_len),
                "second_len": int(second_len),
                "first_color": int(first_color),
                "second_color": int(second_color),
            }
        )

    specs.sort(key=lambda spec: int(spec["outer_len"]), reverse=True)
    return specs


def _sample_shape(specs: list[dict[str, int | str]]) -> tuple[int, int] | None:
    max_side = max(2 * int(spec["outer_len"]) - 1 for spec in specs)
    packed_area = sum((2 * int(spec["outer_len"]) - 1 + 2 * PLACEMENT_PADDING) ** 2 for spec in specs)
    min_side = max(max_side + 2, int(np.ceil(np.sqrt(packed_area))))

    h_lo = max(GRID_HEIGHT_RANGE[0], min_side)
    w_lo = max(GRID_WIDTH_RANGE[0], min_side)
    if h_lo > GRID_HEIGHT_RANGE[1] or w_lo > GRID_WIDTH_RANGE[1]:
        return None
    return random.randint(h_lo, GRID_HEIGHT_RANGE[1]), random.randint(w_lo, GRID_WIDTH_RANGE[1])


def _place_specs(shape: tuple[int, int], specs: list[dict[str, int | str]]) -> bool:
    h, w = shape
    reserved = np.zeros((h, w), dtype=bool)

    for spec in specs:
        outer_len = int(spec["outer_len"])
        radius = outer_len - 1
        if 2 * radius + 1 > h or 2 * radius + 1 > w:
            return False

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            center_r = random.randint(radius, h - radius - 1)
            center_c = random.randint(radius, w - radius - 1)

            r0 = max(0, center_r - radius - PLACEMENT_PADDING)
            r1 = min(h - 1, center_r + radius + PLACEMENT_PADDING)
            c0 = max(0, center_c - radius - PLACEMENT_PADDING)
            c1 = min(w - 1, center_c + radius + PLACEMENT_PADDING)
            if bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any()):
                continue

            reserved[r0 : r1 + 1, c0 : c1 + 1] = True
            spec["center_r"] = int(center_r)
            spec["center_c"] = int(center_c)
            placed = True
            break

        if not placed:
            return False

    return True


def _draw_bar(grid: np.ndarray, spec: dict[str, int | str]) -> None:
    center_r = int(spec["center_r"])
    center_c = int(spec["center_c"])
    first_len = int(spec["first_len"])
    second_len = int(spec["second_len"])
    total = first_len + second_len
    assert total % 2 == 1, "generator keeps an integer midpoint"

    offsets = range(-(total // 2), total // 2 + 1)
    colors = [int(spec["first_color"])] * first_len + [int(spec["second_color"])] * second_len
    assert len(colors) == total

    if spec["orientation"] == "h":
        for dc, color in zip(offsets, colors):
            grid[center_r, center_c + dc] = color
    else:
        for dr, color in zip(offsets, colors):
            grid[center_r + dr, center_c] = color


def _expected_from_specs(shape: tuple[int, int], specs: list[dict[str, int | str]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        center_r = int(spec["center_r"])
        center_c = int(spec["center_c"])
        if int(spec["first_len"]) <= int(spec["second_len"]):
            inner_color, outer_color = int(spec["first_color"]), int(spec["second_color"])
        else:
            inner_color, outer_color = int(spec["second_color"]), int(spec["first_color"])
        _paint_square(out, center_r, center_c, int(spec["outer_len"]), outer_color)
        _paint_square(out, center_r, center_c, int(spec["inner_len"]), inner_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_bar_specs()
    shape = _sample_shape(specs)
    if shape is None or not _place_specs(shape, specs):
        return None

    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        _draw_bar(grid, spec)

    expected = _expected_from_specs(shape, specs)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
