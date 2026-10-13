from __future__ import annotations

from collections import Counter
from functools import lru_cache
import itertools
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_SIDE_RANGE = (14, 28)
GROUP_COUNT_RANGE = (2, 4)
MAX_SIDE_PIXELS = 4
MAX_TOTAL_PIXELS = 10
ENVELOPE_PADDING = 1
GROUP_PLACEMENT_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    """Return the sorted colors present in the grid, optionally excluding some."""
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array in (r,c) order."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _analyze_group(
    grid: np.ndarray,
    color: int,
    *,
    bg: int,
    require_isolated: bool,
) -> dict[str, object]:
    coords = np.argwhere(grid == color)
    assert coords.shape[0] >= 4, "each color group must contain at least four pixels"

    if require_isolated:
        color_mask = grid == color
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            assert sum(
                0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and color_mask[rr, cc]
                for rr, cc in _neighbors4(r, c)
            ) == 0, "input pixels of one color must be isolated singletons"

    row_counts = Counter(map(int, coords[:, 0]))
    col_counts = Counter(map(int, coords[:, 1]))
    row_candidates = [r for r, count in row_counts.items() if count > 1]
    col_candidates = [c for c, count in col_counts.items() if count > 1]
    assert len(row_candidates) == 1, "each color needs one repeated row"
    assert len(col_candidates) == 1, "each color needs one repeated column"

    center_r = row_candidates[0]
    center_c = col_candidates[0]
    assert int(grid[center_r, center_c]) == bg, "the hidden center must stay empty"
    assert np.all(
        (coords[:, 0] == center_r) | (coords[:, 1] == center_c)
    ), "every pixel must lie on the hidden row or hidden column"

    left = sorted(int(c) for r, c in coords if int(r) == center_r and int(c) < center_c)
    right = sorted(int(c) for r, c in coords if int(r) == center_r and int(c) > center_c)
    up = sorted(int(r) for r, c in coords if int(c) == center_c and int(r) < center_r)
    down = sorted(int(r) for r, c in coords if int(c) == center_c and int(r) > center_r)

    assert len(left) + len(right) >= 2, "the hidden row must contain at least two pixels"
    assert len(up) + len(down) >= 2, "the hidden column must contain at least two pixels"
    assert len(left) + len(right) == row_counts[center_r]
    assert len(up) + len(down) == col_counts[center_c]

    return {
        "color": int(color),
        "center": (int(center_r), int(center_c)),
        "left": tuple(left),
        "right": tuple(right),
        "up": tuple(up),
        "down": tuple(down),
    }


def _paint_group(
    out: np.ndarray,
    color: int,
    center: tuple[int, int],
    left_count: int,
    right_count: int,
    up_count: int,
    down_count: int,
    *,
    bg: int,
) -> None:
    r, c = center
    targets = []
    targets.extend((r, c - k) for k in range(1, left_count + 1))
    targets.extend((r, c + k) for k in range(1, right_count + 1))
    targets.extend((r - k, c) for k in range(1, up_count + 1))
    targets.extend((r + k, c) for k in range(1, down_count + 1))

    for rr, cc in targets:
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "compressed pixels must stay in bounds"
        assert int(out[rr, cc]) == bg, "different color groups must not overlap after compression"
        out[rr, cc] = int(color)


def _render_output(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        _paint_group(
            out,
            int(spec["color"]),
            tuple(spec["center"]),
            len(spec["left_in"]),
            len(spec["right_in"]),
            len(spec["up_in"]),
            len(spec["down_in"]),
            bg=BG,
        )
    return out


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        r, c = map(int, spec["center"])
        for dist in spec["left_in"]:
            grid[r, c - int(dist)] = color
        for dist in spec["right_in"]:
            grid[r, c + int(dist)] = color
        for dist in spec["up_in"]:
            grid[r - int(dist), c] = color
        for dist in spec["down_in"]:
            grid[r + int(dist), c] = color
    return grid


def _validate_scene(grid: np.ndarray, *, require_isolated: bool) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30
    bg = get_bg(g)
    assert bg == BG, "generated scenes use black background"
    analyses = [
        _analyze_group(g, color, bg=bg, require_isolated=require_isolated)
        for color in colors_present(g, exclude={bg})
    ]
    assert analyses, "scene must contain at least one color group"
    return analyses


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each color's scattered row/column pixels toward their unseen intersection, leaving the center empty.

    Input structure
    - The background is black.
    - Each foreground color forms one set of isolated pixels.
    - For each color there is a unique hidden center (r,c): every pixel of that color lies either on row r or on column c, the center cell itself is background, and that row and column each contain at least two pixels of the color.

    Rule
    - For each color, identify the unique repeated row and the unique repeated column and use their intersection as the hidden center.
    - Slide the pixels on each of the four rays inward along their own row or column.
    - On one side of the center, preserve the ray order and pack the pixels contiguously so the innermost pixel touches the center.
    - Keep the hidden center as background and process each color independently.

    Key insight
    - The attractor is never drawn: it is reconstructed from the only row and column that each color occupies multiple times, then every sparse ray is compressed toward that empty crossing.
    """
    g = np.asarray(grid, dtype=int)
    analyses = _validate_scene(g, require_isolated=True)

    out = np.full_like(g, BG)
    for info in analyses:
        _paint_group(
            out,
            int(info["color"]),
            tuple(info["center"]),
            len(info["left"]),
            len(info["right"]),
            len(info["up"]),
            len(info["down"]),
            bg=BG,
        )
    return out


@lru_cache(maxsize=None)
def _ray_position_choices(capacity: int, count: int) -> tuple[tuple[int, ...], ...]:
    if count == 0:
        return ((),)
    choices = []
    for combo in itertools.combinations(range(1, capacity + 1), count):
        if all(b - a >= 2 for a, b in zip(combo, combo[1:])):
            choices.append(tuple(map(int, combo)))
    return tuple(choices)


@lru_cache(maxsize=None)
def _side_count_choices(left_cap: int, right_cap: int, up_cap: int, down_cap: int) -> tuple[tuple[int, int, int, int], ...]:
    max_left = min(MAX_SIDE_PIXELS, (left_cap + 1) // 2)
    max_right = min(MAX_SIDE_PIXELS, (right_cap + 1) // 2)
    max_up = min(MAX_SIDE_PIXELS, (up_cap + 1) // 2)
    max_down = min(MAX_SIDE_PIXELS, (down_cap + 1) // 2)

    combos = []
    for left in range(max_left + 1):
        for right in range(max_right + 1):
            if left + right < 2:
                continue
            for up in range(max_up + 1):
                for down in range(max_down + 1):
                    if up + down < 2:
                        continue
                    total = left + right + up + down
                    if total > MAX_TOTAL_PIXELS:
                        continue
                    combos.append((left, right, up, down))
    return tuple(combos)


def _weighted_choice(combos: tuple[tuple[int, int, int, int], ...]) -> tuple[int, int, int, int]:
    weights = []
    for combo in combos:
        zero_sides = sum(v == 0 for v in combo)
        long_arm = max(combo)
        weight = 1.0 + 0.45 * zero_sides + 0.20 * (long_arm >= 3)
        weights.append(weight)
    return random.choices(list(combos), weights=weights, k=1)[0]


def _padded_bbox(coords: list[tuple[int, int]], shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    h, w = shape
    return max(0, r0 - padding), max(0, c0 - padding), min(h - 1, r1 + padding), min(w - 1, c1 + padding)


def _group_spec(shape: tuple[int, int], reserved: np.ndarray, color: int) -> dict[str, object] | None:
    h, w = shape
    for _ in range(GROUP_PLACEMENT_ATTEMPTS):
        center_r = random.randint(1, h - 2)
        center_c = random.randint(1, w - 2)
        left_cap = center_c
        right_cap = w - 1 - center_c
        up_cap = center_r
        down_cap = h - 1 - center_r

        combos = _side_count_choices(left_cap, right_cap, up_cap, down_cap)
        if not combos:
            continue
        left_n, right_n, up_n, down_n = _weighted_choice(combos)

        left_in = random.choice(_ray_position_choices(left_cap, left_n))
        right_in = random.choice(_ray_position_choices(right_cap, right_n))
        up_in = random.choice(_ray_position_choices(up_cap, up_n))
        down_in = random.choice(_ray_position_choices(down_cap, down_n))

        coords = [(center_r, center_c)]
        coords.extend((center_r, center_c - d) for d in left_in)
        coords.extend((center_r, center_c + d) for d in right_in)
        coords.extend((center_r - d, center_c) for d in up_in)
        coords.extend((center_r + d, center_c) for d in down_in)
        coords.extend((center_r, center_c - k) for k in range(1, left_n + 1))
        coords.extend((center_r, center_c + k) for k in range(1, right_n + 1))
        coords.extend((center_r - k, center_c) for k in range(1, up_n + 1))
        coords.extend((center_r + k, center_c) for k in range(1, down_n + 1))

        r0, c0, r1, c1 = _padded_bbox(coords, shape, ENVELOPE_PADDING)
        if np.any(reserved[r0 : r1 + 1, c0 : c1 + 1]):
            continue

        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        return {
            "color": int(color),
            "center": (int(center_r), int(center_c)),
            "left_in": tuple(map(int, left_in)),
            "right_in": tuple(map(int, right_in)),
            "up_in": tuple(map(int, up_in)),
            "down_in": tuple(map(int, down_in)),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    max_groups = min(GROUP_COUNT_RANGE[1], 4 if min(h, w) >= 20 else 3)
    n_groups = random.randint(GROUP_COUNT_RANGE[0], max_groups)
    colors = random.sample(PALETTE, n_groups)

    reserved = np.zeros((h, w), dtype=bool)
    specs = []
    for color in colors:
        spec = _group_spec((h, w), reserved, color)
        if spec is None:
            return None
        specs.append(spec)

    inp = _render_input((h, w), specs)
    expected = _render_output((h, w), specs)

    try:
        analyses = _validate_scene(inp, require_isolated=True)
    except AssertionError:
        return None

    # Independent semantic check: the generated counts must match what the input encodes.
    counts_by_color = {
        int(info["color"]): (len(info["left"]), len(info["right"]), len(info["up"]), len(info["down"]))
        for info in analyses
    }
    for spec in specs:
        encoded = (
            len(spec["left_in"]),
            len(spec["right_in"]),
            len(spec["up_in"]),
            len(spec["down_in"]),
        )
        if counts_by_color[int(spec["color"])] != encoded:
            return None

    if np.array_equal(inp, expected):
        return None
    if get_bg(inp) != BG or get_bg(expected) != BG:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-intersection input")
