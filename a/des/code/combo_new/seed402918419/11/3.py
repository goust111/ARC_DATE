from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MIN = 12
GRID_MAX_EXCLUSIVE = 25
MIN_SEGMENT_LENGTH = 4
MAX_SEGMENT_LENGTH = 11
MAX_GENERATION_ATTEMPTS = 400

BASE_OBSTACLE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair with a straight line that dyes both empty cells and grey obstacles.

    Input structure:
    - The background is black.
    - Grey cells form static obstacle blocks that may or may not lie on future connector paths.
    - Every non-black, non-grey color appears exactly twice, and the two cells of a color lie in the same row or the same column.

    Rule:
    - Read colors by the row-major position of their first visible endpoint in the input.
    - For each color, draw the inclusive straight segment between its two endpoints.
    - Recolor every black or grey cell on that segment to the endpoint color while keeping the endpoints themselves.
    - Grey cells not covered by any segment stay grey, and later segments overwrite earlier ones at crossings.

    Key insight:
    - The colored pixels are only endpoint markers: the real object for each color is the full row or column segment between them, and grey walls are paintable material rather than blockers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for segment in _extract_segments(g):
        _draw_segment(out, segment)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")


def _extract_segments(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    seen = {BLACK, GREY}
    colors: list[int] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in seen:
                seen.add(color)
                colors.append(color)

    assert colors, "expected at least one colored pair"

    segments: list[dict[str, int]] = []
    for color in colors:
        pts = np.argwhere(grid == color)
        assert pts.shape == (2, 2), f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = [tuple(map(int, pt)) for pt in pts]
        assert (r0 == r1) ^ (c0 == c1), f"color {color} endpoints must share one axis"
        if r0 == r1:
            c0, c1 = sorted((c0, c1))
        else:
            r0, r1 = sorted((r0, r1))
        segments.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    return segments


def _draw_segment(canvas: np.ndarray, segment: dict[str, int]) -> None:
    color = int(segment["color"])
    r0 = int(segment["r0"])
    c0 = int(segment["c0"])
    r1 = int(segment["r1"])
    c1 = int(segment["c1"])
    if r0 == r1:
        canvas[r0, c0 : c1 + 1] = color
    else:
        canvas[r0 : r1 + 1, c0] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(GRID_MIN, GRID_MAX_EXCLUSIVE))
    width = int(rng.integers(GRID_MIN, GRID_MAX_EXCLUSIVE))
    color_count = int(rng.integers(2, 6))
    colors = [int(c) for c in rng.choice(COLOR_POOL, size=color_count, replace=False)]

    mode = "cross" if float(rng.random()) < 0.5 else str(rng.choice(["parallel_h", "parallel_v", "mixed"]))
    segments = _build_segments(rng, height, width, colors, mode)
    if segments is None:
        return None

    segment_cells = set().union(*(_segment_cells(seg) for seg in segments))
    endpoint_cells = set().union(*(_endpoint_cells(seg) for seg in segments))
    path_cells = segment_cells - endpoint_cells
    if not path_cells:
        return None

    obstacle_cells: set[tuple[int, int]] = set()

    off_path_target = int(rng.integers(1, 4))
    for _ in range(off_path_target):
        block = _try_place_off_path_block(rng, height, width, obstacle_cells, segment_cells)
        if block is not None:
            obstacle_cells |= block
    if not (obstacle_cells - segment_cells):
        return None

    want_path_obstacle = mode == "cross" or float(rng.random()) < 0.6
    if want_path_obstacle:
        path_block = _try_place_path_block(rng, height, width, obstacle_cells, endpoint_cells, path_cells)
        if path_block is not None:
            obstacle_cells |= path_block
        else:
            fallback = _fallback_path_obstacle(path_cells, obstacle_cells)
            if fallback is None:
                return None
            obstacle_cells |= fallback

    if want_path_obstacle and not (obstacle_cells & path_cells):
        return None

    inp = np.full((height, width), BLACK, dtype=int)
    for r, c in obstacle_cells:
        inp[r, c] = GREY
    for seg in segments:
        inp[seg["r0"], seg["c0"]] = seg["color"]
        inp[seg["r1"], seg["c1"]] = seg["color"]

    if any(int(inp[r, c]) == GREY for r, c in endpoint_cells):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None

    expected = _expected_output(inp, segments)
    if not np.array_equal(solved, expected):
        return None
    if max(inp.shape) > 30 or max(solved.shape) > 30:
        return None
    if len(_extract_segments(inp)) != color_count:
        return None
    return inp


def _build_segments(
    rng: np.random.Generator, height: int, width: int, colors: list[int], mode: str
) -> list[dict[str, int]] | None:
    segments: list[dict[str, int]] = []

    if mode == "cross":
        cross_pair = _make_crossing_pair(rng, height, width, colors[0], colors[1])
        if cross_pair is None:
            return None
        segments.extend(cross_pair)
        remaining = colors[2:]
        orientations = [str(rng.choice(["h", "v"])) for _ in remaining]
    elif mode == "parallel_h":
        remaining = colors
        orientations = ["h"] * len(remaining)
    elif mode == "parallel_v":
        remaining = colors
        orientations = ["v"] * len(remaining)
    else:
        remaining = colors
        h_count = int(rng.integers(1, len(remaining)))
        orientations = ["h"] * h_count + ["v"] * (len(remaining) - h_count)
        orientations = list(rng.permutation(orientations))

    for color, orient in zip(remaining, orientations):
        seg = _sample_nonintersecting_segment(rng, height, width, color, orient, segments)
        if seg is None:
            return None
        segments.append(seg)

    return segments


def _make_crossing_pair(
    rng: np.random.Generator, height: int, width: int, color_h: int, color_v: int
) -> list[dict[str, int]] | None:
    max_h_arm = min(5, width // 2)
    max_v_arm = min(5, height // 2)
    if max_h_arm < 2 or max_v_arm < 2:
        return None

    row = int(rng.integers(2, height - 2))
    col = int(rng.integers(2, width - 2))

    left_extent = int(rng.integers(2, min(max_h_arm, col) + 1))
    right_extent = int(rng.integers(2, min(max_h_arm, width - 1 - col) + 1))
    up_extent = int(rng.integers(2, min(max_v_arm, row) + 1))
    down_extent = int(rng.integers(2, min(max_v_arm, height - 1 - row) + 1))

    horizontal = {
        "color": color_h,
        "r0": row,
        "c0": col - left_extent,
        "r1": row,
        "c1": col + right_extent,
    }
    vertical = {
        "color": color_v,
        "r0": row - up_extent,
        "c0": col,
        "r1": row + down_extent,
        "c1": col,
    }
    return [horizontal, vertical]


def _sample_nonintersecting_segment(
    rng: np.random.Generator,
    height: int,
    width: int,
    color: int,
    orient: str,
    existing: list[dict[str, int]],
) -> dict[str, int] | None:
    existing_endpoints = set().union(*(_endpoint_cells(seg) for seg in existing)) if existing else set()
    for _ in range(160):
        segment = _random_segment(rng, height, width, color, orient)
        if _endpoint_cells(segment) & existing_endpoints:
            continue
        if any(_segments_intersect(segment, other) for other in existing):
            continue
        return segment
    return None


def _random_segment(
    rng: np.random.Generator, height: int, width: int, color: int, orient: str
) -> dict[str, int]:
    if orient == "h":
        length = int(rng.integers(MIN_SEGMENT_LENGTH, min(MAX_SEGMENT_LENGTH, width) + 1))
        row = int(rng.integers(0, height))
        c0 = int(rng.integers(0, width - length + 1))
        c1 = c0 + length - 1
        return {"color": color, "r0": row, "c0": c0, "r1": row, "c1": c1}

    length = int(rng.integers(MIN_SEGMENT_LENGTH, min(MAX_SEGMENT_LENGTH, height) + 1))
    col = int(rng.integers(0, width))
    r0 = int(rng.integers(0, height - length + 1))
    r1 = r0 + length - 1
    return {"color": color, "r0": r0, "c0": col, "r1": r1, "c1": col}


def _segments_intersect(a: dict[str, int], b: dict[str, int]) -> bool:
    a_horizontal = int(a["r0"]) == int(a["r1"])
    b_horizontal = int(b["r0"]) == int(b["r1"])

    if a_horizontal and b_horizontal:
        if int(a["r0"]) != int(b["r0"]):
            return False
        return not (int(a["c1"]) < int(b["c0"]) or int(b["c1"]) < int(a["c0"]))

    if (not a_horizontal) and (not b_horizontal):
        if int(a["c0"]) != int(b["c0"]):
            return False
        return not (int(a["r1"]) < int(b["r0"]) or int(b["r1"]) < int(a["r0"]))

    horizontal = a if a_horizontal else b
    vertical = b if a_horizontal else a
    return (
        int(horizontal["c0"]) <= int(vertical["c0"]) <= int(horizontal["c1"])
        and int(vertical["r0"]) <= int(horizontal["r0"]) <= int(vertical["r1"])
    )


def _endpoint_cells(segment: dict[str, int]) -> set[tuple[int, int]]:
    return {
        (int(segment["r0"]), int(segment["c0"])),
        (int(segment["r1"]), int(segment["c1"])),
    }


def _segment_cells(segment: dict[str, int]) -> set[tuple[int, int]]:
    if int(segment["r0"]) == int(segment["r1"]):
        r = int(segment["r0"])
        return {(r, c) for c in range(int(segment["c0"]), int(segment["c1"]) + 1)}
    c = int(segment["c0"])
    return {(r, c) for r in range(int(segment["r0"]), int(segment["r1"]) + 1)}


def _expected_output(grid: np.ndarray, segments: list[dict[str, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for segment in sorted(segments, key=_first_visible_endpoint):
        _draw_segment(out, segment)
    return out


def _first_visible_endpoint(segment: dict[str, int]) -> tuple[int, int]:
    a = (int(segment["r0"]), int(segment["c0"]))
    b = (int(segment["r1"]), int(segment["c1"]))
    return min(a, b)


def _try_place_off_path_block(
    rng: np.random.Generator,
    height: int,
    width: int,
    obstacle_cells: set[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    for _ in range(80):
        shape = _random_shape_variant(rng)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        base_r = int(rng.integers(0, height - max_r))
        base_c = int(rng.integers(0, width - max_c))
        cells = {(base_r + r, base_c + c) for r, c in shape}
        if cells & obstacle_cells:
            continue
        if cells & forbidden_cells:
            continue
        return cells
    return None


def _try_place_path_block(
    rng: np.random.Generator,
    height: int,
    width: int,
    obstacle_cells: set[tuple[int, int]],
    endpoint_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    path_list = list(path_cells)
    for _ in range(120):
        anchor_target = path_list[int(rng.integers(0, len(path_list)))]
        shape = _random_shape_variant(rng)
        anchor = shape[int(rng.integers(0, len(shape)))]
        delta_r = anchor_target[0] - anchor[0]
        delta_c = anchor_target[1] - anchor[1]
        cells = {(delta_r + r, delta_c + c) for r, c in shape}
        if any(r < 0 or r >= height or c < 0 or c >= width for r, c in cells):
            continue
        if cells & obstacle_cells:
            continue
        if cells & endpoint_cells:
            continue
        if not (cells & path_cells):
            continue
        if not (cells - path_cells):
            continue
        return cells
    return None


def _fallback_path_obstacle(
    path_cells: set[tuple[int, int]], obstacle_cells: set[tuple[int, int]]
) -> set[tuple[int, int]] | None:
    for cell in sorted(path_cells):
        if cell not in obstacle_cells:
            return {cell}
    return None


def _random_shape_variant(rng: np.random.Generator) -> list[tuple[int, int]]:
    base = BASE_OBSTACLE_SHAPES[int(rng.integers(0, len(BASE_OBSTACLE_SHAPES)))]
    variants = _shape_variants(base)
    return variants[int(rng.integers(0, len(variants)))]


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[list[tuple[int, int]]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for swap_axes in (False, True):
        for sign_r in (-1, 1):
            for sign_c in (-1, 1):
                coords = []
                for r, c in shape:
                    rr, cc = (c, r) if swap_axes else (r, c)
                    coords.append((sign_r * rr, sign_c * cc))
                normalized = _normalize_shape(coords)
                variants.add(tuple(normalized))
    return [list(variant) for variant in sorted(variants)]


def _normalize_shape(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)
