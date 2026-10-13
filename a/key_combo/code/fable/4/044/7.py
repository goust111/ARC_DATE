from __future__ import annotations

from collections import Counter

import numpy as np


RED = 2
GREEN = 3
GREY = 5
FOREGROUND_COLORS = [0, 1, 4, 6, 7, 8, 9]
PAIR_SETS = [
    ((3, 2), (4, 2)),
    ((3, 2), (5, 2)),
    ((4, 2), (5, 2)),
    ((3, 3), (4, 2)),
    ((3, 3), (5, 2)),
]
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _find_marker_bars(grid: np.ndarray, color: int) -> list[dict]:
    bars = []
    for coords in _components(grid == color):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != r1:
            continue
        bars.append(
            {
                "color": int(color),
                "row": int(r0),
                "c0": int(c0),
                "c1": int(c1),
                "length": int(c1 - c0 + 1),
                "coords": coords,
            }
        )
    bars.sort(key=lambda item: (item["row"], item["c0"]))
    return bars


def _extract_lane_sprites(grid: np.ndarray, bar: dict, bg: int) -> dict:
    lane = grid[:, bar["c0"] : bar["c1"] + 1].copy()
    lane[bar["row"], :] = bg
    forbidden = {bg, RED, GREEN}
    mask = ~np.isin(lane, list(forbidden))
    comps = _components(mask)
    assert comps, "red lane must contain at least one sprite"

    entries = []
    for coords in comps:
        colors = {int(lane[r, c]) for r, c in coords.tolist()}
        assert len(colors) == 1, "every sprite must be monochrome"
        patch = _crop_mask(coords)
        r0, c0, r1, c1 = _bbox(coords)
        entries.append(
            {
                "coords": coords,
                "color": int(next(iter(colors))),
                "patch": patch,
                "bbox": (r0, c0, r1, c1),
                "top": int(r0),
                "left": int(c0),
            }
        )

    entries.sort(key=lambda item: (item["top"], item["left"]))
    ref_patch = entries[0]["patch"]
    ref_left = entries[0]["left"]
    for entry in entries[1:]:
        assert np.array_equal(entry["patch"], ref_patch), "sprites in one lane must share the same shape"
        assert entry["left"] == ref_left, "sprites in one lane must share the same horizontal offset"

    return {
        "count": len(entries),
        "patch": ref_patch,
        "color": int(entries[0]["color"]),
        "left_offset": int(ref_left),
        "tops": [int(entry["top"]) for entry in entries],
        "coords_list": [entry["coords"] for entry in entries],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each red-lane sprite set into the uniquely divisible green lane, scaling by the bar-length ratio.

    Input structure:
    - The background is the most common color.
    - Thin horizontal red and green bars mark vertical lanes of distinct widths.
    - Each red lane contains repeated identical monochrome sprites, while green lanes are empty.

    Rule:
    - Pair every red bar with the unique green bar whose length is an integer multiple of the red length.
    - Turn the source sprites grey in place.
    - In the paired green lane, scale the source sprite by the length ratio, recolor it green, stack the same number of copies bottom-up with equal k-row gaps, and remove all marker bars.

    Key insight:
    - The divisibility ratio between bar lengths simultaneously identifies the destination lane and the sprite scaling factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    red_bars = _find_marker_bars(g, RED)
    green_bars = _find_marker_bars(g, GREEN)
    assert red_bars and green_bars, "expected both red and green marker bars"

    green_by_length = {bar["length"]: bar for bar in green_bars}
    assert len(green_by_length) == len(green_bars), "green bar lengths must be unique"
    assert len({bar["length"] for bar in red_bars}) == len(red_bars), "red bar lengths must be unique"

    out = g.copy()

    for red_bar in red_bars:
        lane_info = _extract_lane_sprites(g, red_bar, bg)
        candidates = [bar for bar in green_bars if bar["length"] % red_bar["length"] == 0]
        assert len(candidates) == 1, "every red bar must match exactly one green bar"
        green_bar = candidates[0]
        factor = green_bar["length"] // red_bar["length"]
        assert factor >= 1

        for coords in lane_info["coords_list"]:
            out[coords[:, 0], red_bar["c0"] + coords[:, 1]] = GREY

        scaled = _scale_mask(lane_info["patch"], factor)
        scaled_h, scaled_w = scaled.shape
        left = green_bar["c0"] + lane_info["left_offset"] * factor
        assert left + scaled_w <= green_bar["c1"] + 1, "scaled sprite must fit inside destination lane"
        step = scaled_h + factor

        for idx in range(lane_info["count"]):
            top = h - scaled_h - idx * step
            assert top >= 0, "scaled stack must fit vertically"
            target = out[top : top + scaled_h, left : left + scaled_w]
            assert target.shape == scaled.shape
            target[scaled] = GREEN

    for bar in red_bars + green_bars:
        out[bar["coords"][:, 0], bar["coords"][:, 1]] = bg

    return out


def _random_connected_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    min_area: int,
    max_area: int,
) -> np.ndarray:
    area = int(rng.integers(min_area, max_area + 1))
    start = (int(rng.integers(height)), int(rng.integers(width)))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        choice = frontier[int(rng.integers(len(frontier)))]
        cells.add(choice)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    patch = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True
    return patch


def _sample_pair_spec(rng: np.random.Generator, red_len: int, factor: int, color: int) -> dict:
    sprite_w = int(rng.integers(2, min(3, red_len) + 1))
    sprite_h = int(rng.integers(2, 4))
    max_area = sprite_h * sprite_w
    min_area = max(2, max_area // 2)
    patch = _random_connected_mask(rng, sprite_h, sprite_w, min_area, max_area)
    left_offset = int(rng.integers(0, red_len - patch.shape[1] + 1))

    max_count = 3
    while max_count > 1:
        src_h = max_count * patch.shape[0] + (max_count - 1)
        dst_h = max_count * (patch.shape[0] * factor) + (max_count - 1) * factor
        if max(src_h, dst_h) + 2 <= 30:
            break
        max_count -= 1
    count = int(rng.integers(1, max_count + 1))

    return {
        "red_len": int(red_len),
        "green_len": int(red_len * factor),
        "factor": int(factor),
        "patch": patch,
        "left_offset": int(left_offset),
        "color": int(color),
        "count": int(count),
    }


def _stack_height(patch_h: int, count: int, gap: int) -> int:
    return count * patch_h + (count - 1) * gap


def _paint_stack(
    grid: np.ndarray,
    patch: np.ndarray,
    top_positions: list[int],
    left: int,
    color: int,
) -> None:
    ph, pw = patch.shape
    for top in top_positions:
        view = grid[top : top + ph, left : left + pw]
        view[patch] = color


def _build_expected(
    pair_specs: list[dict],
    lane_positions: dict[tuple[int, str], int],
    shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    for pair_idx, spec in enumerate(pair_specs):
        red_left = lane_positions[(pair_idx, "R")]
        green_left = lane_positions[(pair_idx, "G")]
        red_tops = [
            h - spec["patch"].shape[0] - idx * (spec["patch"].shape[0] + 1)
            for idx in range(spec["count"])
        ]
        _paint_stack(out, spec["patch"], red_tops, red_left + spec["left_offset"], GREY)

        scaled = _scale_mask(spec["patch"], spec["factor"])
        scaled_tops = [
            h - scaled.shape[0] - idx * (scaled.shape[0] + spec["factor"])
            for idx in range(spec["count"])
        ]
        _paint_stack(
            out,
            scaled,
            scaled_tops,
            green_left + spec["left_offset"] * spec["factor"],
            GREEN,
        )
    return out


def _build_input(pair_specs: list[dict], lane_positions: dict[tuple[int, str], int], shape: tuple[int, int], bg: int) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), bg, dtype=int)
    for pair_idx, spec in enumerate(pair_specs):
        red_left = lane_positions[(pair_idx, "R")]
        green_left = lane_positions[(pair_idx, "G")]
        grid[0, red_left : red_left + spec["red_len"]] = RED
        grid[0, green_left : green_left + spec["green_len"]] = GREEN

        tops = [
            h - spec["patch"].shape[0] - idx * (spec["patch"].shape[0] + 1)
            for idx in range(spec["count"])
        ]
        _paint_stack(grid, spec["patch"], tops, red_left + spec["left_offset"], spec["color"])
    return grid


def _nearest_green_is_pair(
    lane_positions: dict[tuple[int, str], int],
    pair_specs: list[dict],
) -> bool:
    green_centers = {
        pair_idx: lane_positions[(pair_idx, "G")] + (pair_specs[pair_idx]["green_len"] - 1) / 2.0
        for pair_idx in range(len(pair_specs))
    }
    for pair_idx in range(len(pair_specs)):
        red_center = lane_positions[(pair_idx, "R")] + (pair_specs[pair_idx]["red_len"] - 1) / 2.0
        nearest = min(green_centers, key=lambda idx: abs(green_centers[idx] - red_center))
        if nearest != pair_idx:
            return False
    return True


def _choose_layout(
    rng: np.random.Generator,
    pair_specs: list[dict],
) -> tuple[int, dict[tuple[int, str], int]] | None:
    order_base = [(0, "R", pair_specs[0]["red_len"]), (0, "G", pair_specs[0]["green_len"]), (1, "R", pair_specs[1]["red_len"]), (1, "G", pair_specs[1]["green_len"])]
    total_width = sum(item[2] for item in order_base)

    for _ in range(80):
        order = [order_base[idx] for idx in rng.permutation(len(order_base)).tolist()]
        gap_budget = 30 - total_width
        if gap_budget < 3:
            return None

        left_pad = int(rng.integers(0, min(2, gap_budget - 3) + 1))
        remaining = gap_budget - left_pad
        gaps = [1, 1, 1]
        remaining -= 3
        for gap_idx in range(3):
            extra_max = min(1, remaining)
            extra = int(rng.integers(0, extra_max + 1))
            gaps[gap_idx] += extra
            remaining -= extra
        right_pad = int(rng.integers(0, remaining + 1))
        width = total_width + left_pad + sum(gaps) + right_pad

        lane_positions: dict[tuple[int, str], int] = {}
        cursor = left_pad
        for idx, (pair_idx, kind, lane_width) in enumerate(order):
            lane_positions[(pair_idx, kind)] = cursor
            cursor += lane_width
            if idx < len(order) - 1:
                cursor += gaps[idx]

        require_non_nearest = bool(rng.integers(0, 2))
        if require_non_nearest and _nearest_green_is_pair(lane_positions, pair_specs):
            continue
        return width, lane_positions
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    pair_set = PAIR_SETS[int(rng.integers(len(PAIR_SETS)))]
    bg = int(rng.choice(FOREGROUND_COLORS))
    sprite_colors = [c for c in FOREGROUND_COLORS if c != bg]
    rng.shuffle(sprite_colors)

    pair_specs = [
        _sample_pair_spec(rng, red_len, factor, sprite_colors[idx])
        for idx, (red_len, factor) in enumerate(pair_set)
    ]

    required_height = 2 + max(
        max(
            _stack_height(spec["patch"].shape[0], spec["count"], 1),
            _stack_height(spec["patch"].shape[0] * spec["factor"], spec["count"], spec["factor"]),
        )
        for spec in pair_specs
    )
    if required_height > 30:
        return None

    extra_height = int(rng.integers(0, 31 - required_height))
    height = required_height + extra_height

    chosen = _choose_layout(rng, pair_specs)
    if chosen is None:
        return None
    width, lane_positions = chosen

    inp = _build_input(pair_specs, lane_positions, (height, width), bg)
    expected = _build_expected(pair_specs, lane_positions, (height, width), bg)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
