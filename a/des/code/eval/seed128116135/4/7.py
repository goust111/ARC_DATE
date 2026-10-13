from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
LINE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 300
EXTRA_LINE_ATTEMPTS = 200
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
SIDES = ("left", "right", "top", "bottom")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor rope crossings according to edge markers that specify over/under order.

    Input structure
    - Background is black.
    - Each non-black color belongs to a single straight 1-pixel-wide line segment:
      horizontal, vertical, or 45-degree diagonal.
    - Small 2x1 vertical markers sit on one grid edge. The top marker pixel is the
      overpassing rope color; the bottom marker pixel is the underpassing rope color.
    - Two rope colors may share one geometric crossing, but only one of the two
      colors is visible at that crossing pixel in the input.

    Rule
    - For each edge marker, find the unique crossing of the two corresponding line
      segments.
    - Color the crossing pixel with the marker's top color.
    - Along the underpassing rope, recolor the immediately previous and next pixels
      around the crossing to grey, simulating occlusion.
    - Erase the edge markers to black.
    """
    g = np.asarray(grid, dtype=int)
    markers, marker_mask = _detect_markers(g)

    out = g.copy()
    out[marker_mask] = BLACK

    colors = sorted({m["over"] for m in markers} | {m["under"] for m in markers})
    lines = {color: _infer_line(g, color, marker_mask) for color in colors}

    for marker in markers:
        over = int(marker["over"])
        under = int(marker["under"])
        inter = lines[over]["set"] & lines[under]["set"]
        assert len(inter) == 1, f"expected one crossing for {over}/{under}, got {sorted(inter)}"
        cross = next(iter(inter))
        out[cross] = over

        under_points = lines[under]["points"]
        idx = lines[under]["index"][cross]
        if idx - 1 >= 0:
            out[under_points[idx - 1]] = GREY
        if idx + 1 < len(under_points):
            out[under_points[idx + 1]] = GREY

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rope-weave input")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 27))
    w = int(rng.integers(18, 27))
    side = str(rng.choice(SIDES))
    marker_capacity = _marker_capacity(side, h, w)
    pair_count = int(rng.integers(1, min(4, marker_capacity, len(LINE_COLORS) // 2) + 1))

    colors = list(LINE_COLORS)
    rng.shuffle(colors)
    pair_colors = [(int(colors[2 * i]), int(colors[2 * i + 1])) for i in range(pair_count)]
    extra_colors = colors[2 * pair_count :]

    bounds = _interior_bounds(side, h, w)
    occupied: set[tuple[int, int]] = set()
    pairs = []

    for color_a, color_b in pair_colors:
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            dir_a, dir_b = _sample_direction_pair(rng)
            cross = _sample_cross(bounds, rng)
            line_a = _sample_line_through(cross, dir_a, bounds, rng)
            line_b = _sample_line_through(cross, dir_b, bounds, rng)
            if line_a is None or line_b is None:
                continue
            if (set(line_a) & set(line_b)) != {cross}:
                continue
            combined = set(line_a) | set(line_b)
            if combined & occupied:
                continue

            over, under = ((color_a, color_b) if bool(rng.integers(0, 2)) else (color_b, color_a))
            visible = int(rng.choice((color_a, color_b)))
            pairs.append(
                {
                    "colors": (color_a, color_b),
                    "cross": cross,
                    "line_a": line_a,
                    "line_b": line_b,
                    "over": int(over),
                    "under": int(under),
                    "visible": visible,
                }
            )
            occupied |= combined
            placed = True
            break
        if not placed:
            return None

    extra_lines = []
    rng.shuffle(extra_colors)
    extra_target = int(rng.integers(0, min(3, len(extra_colors)) + 1))
    for color in extra_colors[:extra_target]:
        placed = False
        for _ in range(EXTRA_LINE_ATTEMPTS):
            direction = DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]
            center = _sample_cross(bounds, rng)
            line = _sample_line_through(center, direction, bounds, rng, min_steps=1)
            if line is None:
                continue
            if set(line) & occupied:
                continue
            extra_lines.append((int(color), line))
            occupied |= set(line)
            placed = True
            break
        if not placed:
            return None

    marker_positions = _sample_marker_positions(side, h, w, pair_count, rng)
    if marker_positions is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)

    for pair in pairs:
        color_a, color_b = pair["colors"]
        cross = pair["cross"]
        if pair["visible"] == color_a:
            _paint_line(grid, pair["line_b"], color_b, skip={cross})
            _paint_line(grid, pair["line_a"], color_a)
        else:
            _paint_line(grid, pair["line_a"], color_a, skip={cross})
            _paint_line(grid, pair["line_b"], color_b)

    for color, line in extra_lines:
        _paint_line(grid, line, color)

    rng.shuffle(pairs)
    for pair, pos in zip(pairs, marker_positions):
        _paint_marker(grid, side, pos, int(pair["over"]), int(pair["under"]))

    try:
        markers, marker_mask = _detect_markers(grid)
        if len(markers) != pair_count:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == GREY) < 2 * pair_count:
        return None
    if not np.all(grid[marker_mask] != BLACK):
        return None
    return grid


def _detect_markers(grid: np.ndarray) -> tuple[list[dict[str, int | tuple[int, int]]], np.ndarray]:
    h, w = grid.shape
    candidates_by_side: dict[str, list[dict[str, int | tuple[int, int]]]] = {side: [] for side in SIDES}

    for coords in _components4(grid != BLACK):
        if coords.shape[0] != 2:
            continue
        rows = coords[:, 0]
        cols = coords[:, 1]
        if int(rows.max() - rows.min()) != 1 or int(cols.max() - cols.min()) != 0:
            continue

        colors = [int(grid[int(r), int(c)]) for r, c in coords]
        if BLACK in colors or colors[0] == colors[1]:
            continue

        touched = []
        if np.any(cols == 0):
            touched.append("left")
        if np.any(cols == w - 1):
            touched.append("right")
        if np.any(rows == 0):
            touched.append("top")
        if np.any(rows == h - 1):
            touched.append("bottom")
        if len(touched) != 1:
            continue

        order = coords[np.argsort(coords[:, 0])]
        marker = {
            "over": int(grid[tuple(order[0])]),
            "under": int(grid[tuple(order[1])]),
            "top": (int(order[0, 0]), int(order[0, 1])),
            "bottom": (int(order[1, 0]), int(order[1, 1])),
        }
        candidates_by_side[touched[0]].append(marker)

    counts = {side: len(markers) for side, markers in candidates_by_side.items()}
    best_count = max(counts.values(), default=0)
    assert best_count > 0, "expected at least one edge marker"
    best_sides = [side for side, count in counts.items() if count == best_count]
    assert len(best_sides) == 1, f"ambiguous marker side counts: {counts}"

    markers = candidates_by_side[best_sides[0]]
    marker_mask = np.zeros_like(grid, dtype=bool)
    for marker in markers:
        marker_mask[marker["top"]] = True
        marker_mask[marker["bottom"]] = True

    for marker in markers:
        for color in (int(marker["over"]), int(marker["under"])):
            assert np.any((grid == color) & ~marker_mask), f"marker color {color} must also appear on a rope"

    return markers, marker_mask


def _infer_line(grid: np.ndarray, color: int, marker_mask: np.ndarray) -> dict[str, object]:
    coords = np.argwhere((grid == int(color)) & ~marker_mask)
    assert coords.shape[0] >= 2, f"color {color} must have at least two rope pixels"

    rows = coords[:, 0]
    cols = coords[:, 1]

    if np.all(rows == rows[0]):
        row = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        points = [(row, c) for c in range(c0, c1 + 1)]
    elif np.all(cols == cols[0]):
        col = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        points = [(r, col) for r in range(r0, r1 + 1)]
    elif np.all(rows - cols == rows[0] - cols[0]):
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols[rows.argmin()])
        points = [(r0 + k, c0 + k) for k in range(r1 - r0 + 1)]
    elif np.all(rows + cols == rows[0] + cols[0]):
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols[rows.argmin()])
        points = [(r0 + k, c0 - k) for k in range(r1 - r0 + 1)]
    else:
        raise AssertionError(f"color {color} is not a supported straight line")

    line_set = set(points)
    assert set(map(tuple, coords)) <= line_set, f"color {color} deviates from its support line"
    return {"points": points, "set": line_set, "index": {pt: i for i, pt in enumerate(points)}}


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))

    return comps


def _marker_capacity(side: str, h: int, w: int) -> int:
    if side in ("left", "right"):
        starts = list(range(1, h - 2))
        gap = 3
    else:
        starts = list(range(1, w - 1))
        gap = 2

    chosen = []
    for start in starts:
        if all(abs(start - other) >= gap for other in chosen):
            chosen.append(start)
    return len(chosen)


def _interior_bounds(side: str, h: int, w: int) -> tuple[int, int, int, int]:
    r_lo, r_hi = 1, h - 2
    c_lo, c_hi = 1, w - 2
    if side == "left":
        c_lo = 3
    elif side == "right":
        c_hi = w - 4
    elif side == "top":
        r_lo = 3
    elif side == "bottom":
        r_hi = h - 4
    return r_lo, r_hi, c_lo, c_hi


def _sample_direction_pair(rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
    idx = rng.choice(len(DIRECTIONS), size=2, replace=False)
    return DIRECTIONS[int(idx[0])], DIRECTIONS[int(idx[1])]


def _sample_cross(bounds: tuple[int, int, int, int], rng: np.random.Generator) -> tuple[int, int]:
    r_lo, r_hi, c_lo, c_hi = bounds
    return int(rng.integers(r_lo, r_hi + 1)), int(rng.integers(c_lo, c_hi + 1))


def _sample_line_through(
    cross: tuple[int, int],
    direction: tuple[int, int],
    bounds: tuple[int, int, int, int],
    rng: np.random.Generator,
    min_steps: int = 2,
) -> list[tuple[int, int]] | None:
    back = _available_steps(cross, direction, bounds, sign=-1)
    forward = _available_steps(cross, direction, bounds, sign=1)
    if back < min_steps or forward < min_steps:
        return None

    before = int(rng.integers(min_steps, min(back, 7) + 1))
    after = int(rng.integers(min_steps, min(forward, 7) + 1))
    dr, dc = direction
    return [(cross[0] + k * dr, cross[1] + k * dc) for k in range(-before, after + 1)]


def _available_steps(
    cross: tuple[int, int],
    direction: tuple[int, int],
    bounds: tuple[int, int, int, int],
    sign: int,
) -> int:
    r_lo, r_hi, c_lo, c_hi = bounds
    dr, dc = direction
    steps = 0
    while True:
        nr = int(cross[0] + (steps + 1) * sign * dr)
        nc = int(cross[1] + (steps + 1) * sign * dc)
        if not (r_lo <= nr <= r_hi and c_lo <= nc <= c_hi):
            return steps
        steps += 1


def _sample_marker_positions(
    side: str,
    h: int,
    w: int,
    count: int,
    rng: np.random.Generator,
) -> list[int] | None:
    if side in ("left", "right"):
        candidates = list(range(1, h - 2))
        gap = 3
    else:
        candidates = list(range(1, w - 1))
        gap = 2

    rng.shuffle(candidates)
    chosen = []
    for pos in candidates:
        if all(abs(pos - other) >= gap for other in chosen):
            chosen.append(int(pos))
        if len(chosen) == count:
            return chosen
    return None


def _paint_line(
    grid: np.ndarray,
    points: list[tuple[int, int]],
    color: int,
    skip: set[tuple[int, int]] | None = None,
) -> None:
    for pt in points:
        if skip is not None and pt in skip:
            continue
        grid[pt] = int(color)


def _paint_marker(grid: np.ndarray, side: str, pos: int, over: int, under: int) -> None:
    h, w = grid.shape
    if side == "left":
        grid[pos, 0] = over
        grid[pos + 1, 0] = under
    elif side == "right":
        grid[pos, w - 1] = over
        grid[pos + 1, w - 1] = under
    elif side == "top":
        grid[0, pos] = over
        grid[1, pos] = under
    elif side == "bottom":
        grid[h - 2, pos] = over
        grid[h - 1, pos] = under
    else:
        raise AssertionError(f"unsupported side {side}")
