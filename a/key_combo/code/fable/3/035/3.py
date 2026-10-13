from __future__ import annotations

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRY_LIMIT = 400


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _assert_solid_rectangle(grid: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    assert int(obj["area"]) == _bbox_h(obj["bbox"]) * _bbox_w(obj["bbox"]), "expected filled rectangles"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "expected monochrome solid rectangles"


def _render_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _parse_scene(grid: np.ndarray) -> tuple[dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g != BG), "expected foreground objects"

    objects = _find_components(g, bg=BG)
    assert len(objects) >= 2, "expected one falling square and at least one bar"
    for obj in objects:
        _assert_solid_rectangle(g, obj)

    top_obj = min(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    assert _bbox_h(top_obj["bbox"]) == _bbox_w(top_obj["bbox"]), "top object must be the falling square"

    square = top_obj
    bars = [obj for obj in objects if obj is not square]
    assert bars, "expected at least one bar"

    sr0, sc0, sr1, sc1 = square["bbox"]
    for bar in bars:
        r0, c0, r1, c1 = bar["bbox"]
        assert r0 > sr1, "bars must lie below the square"
        assert _bbox_w(bar["bbox"]) > _bbox_h(bar["bbox"]), "non-square objects must be horizontal bars"
        assert c0 <= sc1 and c1 >= sc0, "each bar must overlap the square's column span"

    bars.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return square, bars


def _simulate(
    shape: tuple[int, int],
    square: dict,
    bars: list[dict],
    bg: int = BG,
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), int(bg), dtype=int)

    sr0, sc0, sr1, sc1 = square["bbox"]
    side = _bbox_h(square["bbox"])
    stop_top = height - side
    stop_index: int | None = None

    for idx, bar in enumerate(bars):
        _r0, c0, _r1, c1 = bar["bbox"]
        has_left = c0 < sc0
        has_right = c1 > sc1
        if has_left and has_right:
            stop_top = int(bar["bbox"][0] - side)
            stop_index = idx
            break

    for idx, bar in enumerate(bars):
        r0, c0, r1, c1 = bar["bbox"]
        color = int(bar["color"])

        if stop_index is not None and idx >= stop_index:
            _render_rect(out, bar["bbox"], color)
            continue

        left_end = min(c1, sc0 - 1)
        if c0 <= left_end:
            new_bbox = (r0, c0 - 1, r1, left_end - 1)
            assert 0 <= new_bbox[1] <= new_bbox[3] < width, "shifted left bar segment left the canvas"
            _render_rect(out, new_bbox, color)

        right_start = max(c0, sc1 + 1)
        if right_start <= c1:
            new_bbox = (r0, right_start + 1, r1, c1 + 1)
            assert 0 <= new_bbox[1] <= new_bbox[3] < width, "shifted right bar segment left the canvas"
            _render_rect(out, new_bbox, color)

    final_square = (int(stop_top), int(sc0), int(stop_top + side - 1), int(sc1))
    assert 0 <= final_square[0] <= final_square[2] < height, "final square position is out of bounds"
    assert np.all(out[final_square[0] : final_square[2] + 1, final_square[1] : final_square[3] + 1] == bg), (
        "final square must land on an empty region"
    )
    _render_rect(out, final_square, int(square["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the solid square until the first bar with support on both sides catches it.

    Input structure:
    - Black background with one solid colored square near the top.
    - Below it sit several disjoint monochrome solid horizontal bars of different colors.
    - Every bar intersects the square's fixed column span, which defines the falling path.

    Rule:
    - The square falls straight downward without changing columns.
    - When it reaches a bar, the bar supports the square iff it extends beyond the path on both
      the left and right sides; then the square stops directly above that bar and everything below
      remains unchanged.
    - Otherwise the bar is pierced: erase the path-overlapping stripe, shift the left remnant one
      cell left and the right remnant one cell right, and continue falling.
    - If no bar has support on both sides, the square ends flush with the bottom edge.

    Key insight:
    - A bar's length, thickness, and color do not decide the landing point; only whether the
      falling path has support on both sides at that bar matters.
    """
    g = np.asarray(grid, dtype=int)
    square, bars = _parse_scene(g)
    return _simulate(g.shape, square, bars, bg=BG)


def _bar_bbox_for_type(
    rng: np.random.Generator,
    bar_type: str,
    row0: int,
    bar_h: int,
    path_c0: int,
    path_c1: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    side = int(path_c1 - path_c0 + 1)

    if bar_type == "support":
        left_choices = list(range(1, path_c0 + 1))
        right_choices = list(range(1, width - path_c1))
        if not left_choices or not right_choices:
            return None
        left_extra = int(rng.choice(np.array(left_choices, dtype=int)))
        right_extra = int(rng.choice(np.array(right_choices, dtype=int)))
        c0 = path_c0 - left_extra
        c1 = path_c1 + right_extra
    elif bar_type == "left_only":
        left_choices = list(range(1, path_c0))
        if not left_choices:
            return None
        left_extra = int(rng.choice(np.array(left_choices, dtype=int)))
        right_trim = int(rng.integers(0, side))
        c0 = path_c0 - left_extra
        c1 = path_c1 - right_trim
    elif bar_type == "right_only":
        right_choices = list(range(1, width - path_c1 - 1))
        if not right_choices:
            return None
        right_extra = int(rng.choice(np.array(right_choices, dtype=int)))
        left_trim = int(rng.integers(0, side))
        c0 = path_c0 + left_trim
        c1 = path_c1 + right_extra
    elif bar_type == "inside":
        min_w = bar_h + 1
        if side < min_w:
            return None
        span = int(rng.integers(min_w, side + 1))
        offset = int(rng.integers(0, side - span + 1))
        c0 = path_c0 + offset
        c1 = c0 + span - 1
    else:
        return None

    if c0 < 0 or c1 >= width or c0 > c1:
        return None
    if c0 > path_c1 or c1 < path_c0:
        return None
    if c1 - c0 + 1 <= bar_h:
        return None
    return (int(row0), int(c0), int(row0 + bar_h - 1), int(c1))


def _sample_layout(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    side = int(rng.integers(2, 5))
    bar_count = int(rng.integers(2, 6))
    square_top = int(rng.integers(0, 3))
    bar_heights = [int(rng.integers(1, 3)) for _ in range(bar_count)]
    first_gap = int(rng.integers(1, 4))
    gaps = [int(rng.integers(1, 4)) for _ in range(bar_count - 1)]
    tail_gap = int(rng.integers(0, 4))

    height = square_top + side + first_gap + sum(bar_heights) + sum(gaps) + tail_gap
    if not 10 <= height <= 30:
        return None

    min_width = max(10, side + 6)
    width = int(rng.integers(min_width, 31))
    square_left = int(rng.integers(2, width - side - 1))
    path_c0 = square_left
    path_c1 = square_left + side - 1

    has_support = bool(rng.random() < 0.6)
    if has_support:
        if bar_count >= 3 and rng.random() < 0.6:
            support_index = int(rng.integers(1, bar_count))
        else:
            support_index = int(rng.integers(0, bar_count))
    else:
        support_index = None

    bar_rows: list[int] = []
    row = square_top + side + first_gap
    for idx, bar_h in enumerate(bar_heights):
        bar_rows.append(int(row))
        row += bar_h
        if idx < bar_count - 1:
            row += gaps[idx]

    colors = rng.choice(np.arange(1, 10, dtype=int), size=bar_count + 1, replace=False)
    square_color = int(colors[0])
    bar_colors = [int(c) for c in colors[1:]]

    bars: list[dict] = []
    for idx, (row0, bar_h, color) in enumerate(zip(bar_rows, bar_heights, bar_colors)):
        if support_index is not None and idx == support_index:
            kind = "support"
        elif support_index is not None and idx > support_index:
            kind = str(rng.choice(np.array(["support", "left_only", "right_only", "inside"], dtype=object)))
        else:
            kind = str(rng.choice(np.array(["left_only", "right_only", "inside"], dtype=object), p=[0.4, 0.4, 0.2]))

        bbox = _bar_bbox_for_type(rng, kind, row0, bar_h, path_c0, path_c1, width)
        if bbox is None:
            return None
        bars.append({"bbox": bbox, "color": color})

    inp = np.full((height, width), BG, dtype=int)
    square_bbox = (square_top, square_left, square_top + side - 1, square_left + side - 1)
    _render_rect(inp, square_bbox, square_color)
    for bar in bars:
        _render_rect(inp, bar["bbox"], int(bar["color"]))

    square = {"bbox": square_bbox, "color": square_color}
    expected = _simulate(inp.shape, square, bars, bg=BG)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        rng = np.random.default_rng()
        sample = _sample_layout(rng)
        if sample is None:
            continue
        inp, _out = sample
        return inp
    raise RuntimeError("failed to generate a valid falling-square support input")
