from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
AZURE = 8
STAMP_COLOR_POOL = (1, 2, 3, 6, 7, 9)

_RNG = random.Random()
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_canvas_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox(g == GREY)
    assert box is not None, "missing grey canvas"
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "canvas must enclose an interior"
    border = np.zeros_like(g, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    assert np.array_equal(g == GREY, border), "grey pixels must form one rectangular border"
    return r0, c0, r1, c1


def _extract_stamp(g: np.ndarray) -> np.ndarray:
    special = {BG, YELLOW, GREY, AZURE}
    mask = np.isin(g, list(special), invert=True)
    box = _bbox(mask)
    assert box is not None, "missing external stamp"
    r0, c0, r1, c1 = box
    stamp = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = sorted(int(v) for v in np.unique(stamp) if int(v) not in special)
    assert 1 <= len(colors) <= 3, "stamp must use 1-3 non-special colors"
    return stamp


def _trace_path(g: np.ndarray, canvas_bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = canvas_bbox
    mask = np.zeros_like(g, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = (g[r0 + 1 : r1, c0 + 1 : c1] == AZURE) | (
        g[r0 + 1 : r1, c0 + 1 : c1] == YELLOW
    )
    pts = {tuple(map(int, pt)) for pt in np.argwhere(mask)}
    assert pts, "path must exist inside the canvas"

    endpoints = sorted(tuple(map(int, pt)) for pt in np.argwhere(g == YELLOW))
    assert len(endpoints) == 2, "expected exactly two yellow endpoints"
    assert endpoints[0] in pts and endpoints[1] in pts, "yellow endpoints must lie on the path"

    degrees: dict[tuple[int, int], int] = {}
    for p in pts:
        degrees[p] = sum((nr, nc) in pts for nr, nc in _neighbors4(*p))
    assert degrees[endpoints[0]] == 1 and degrees[endpoints[1]] == 1, "path endpoints must have degree 1"
    for p in pts:
        if p not in endpoints:
            assert degrees[p] == 2, "path must be a single non-branching polyline"

    start = endpoints[0]
    end = endpoints[1]
    turns: list[tuple[int, int]] = []
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while cur != end:
        options = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(options) == 1, "path traversal must be unambiguous"
        nxt = options[0]
        if prev is not None:
            d1 = (cur[0] - prev[0], cur[1] - prev[1])
            d2 = (nxt[0] - cur[0], nxt[1] - cur[1])
            if d1 != d2:
                turns.append(cur)
        prev, cur = cur, nxt
        assert cur not in visited or cur == end, "path must not self-intersect"
        visited.add(cur)

    assert visited == pts, "path pixels must form a single chain"
    return start, turns


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= out.shape[0] and left + w <= out.shape[1], "stamp out of bounds"
    out[top : top + h, left : left + w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the external motif onto the canvas at the path start and at each turn,
    rotating the stamp clockwise by 90 degrees after every turn.
    """
    g = np.asarray(grid, dtype=int)
    canvas_bbox = _find_canvas_bbox(g)
    stamp = _extract_stamp(g)
    start, turns = _trace_path(g, canvas_bbox)

    out = g.copy()
    out[(out == AZURE) | (out == YELLOW)] = BG

    placements = [start] + turns
    for i, (r, c) in enumerate(placements):
        patch = np.rot90(stamp, -int(i % 4))
        _paste(out, patch, r, c)

    r0, c0, r1, c1 = canvas_bbox
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    return out


def _all_rotations_distinct_enough(stamp: np.ndarray) -> bool:
    signatures = {np.rot90(stamp, -k).tobytes() + str(np.rot90(stamp, -k).shape).encode() for k in range(4)}
    return len(signatures) >= 2


def _make_stamp(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        n_colors = rng.randint(1, 3)
        colors = list(rng.sample(STAMP_COLOR_POOL, n_colors))
        stamp = np.empty((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                stamp[r, c] = colors[rng.randrange(n_colors)]
        for color in colors:
            if color not in stamp:
                rr = rng.randrange(h)
                cc = rng.randrange(w)
                stamp[rr, cc] = color
        if _all_rotations_distinct_enough(stamp):
            return stamp
    raise RuntimeError("failed to build a usable stamp")


def _spaced_sorted_positions(length: int, count: int, gap: int, rng: random.Random) -> list[int] | None:
    if count <= 0:
        return []
    compressed = length - (count - 1) * (gap - 1)
    if compressed < count:
        return None
    base = sorted(rng.sample(range(compressed), count))
    return [base[i] + i * (gap - 1) for i in range(count)]


def _jagged_sequence(
    length: int,
    count: int,
    gap: int,
    rng: random.Random,
    require_last_not_less: bool = False,
) -> list[int] | None:
    if count <= 0:
        return []
    for _ in range(200):
        seq = [rng.randrange(length)]
        while len(seq) < count:
            choices = [v for v in range(length) if abs(v - seq[-1]) >= gap]
            if not choices:
                break
            seq.append(rng.choice(choices))
        if len(seq) != count:
            continue
        if require_last_not_less and seq[-1] < seq[0]:
            continue
        return seq
    return None


def _generate_polyline(
    safe_h: int,
    safe_w: int,
    turns: int,
    start_axis: str,
    rng: random.Random,
    min_step: int = 2,
) -> list[tuple[int, int]] | None:
    segments = turns + 1
    for _ in range(200):
        if start_axis == "H":
            h_segments = (segments + 1) // 2
            v_segments = segments // 2
            x_levels = _spaced_sorted_positions(safe_w, h_segments + 1, min_step, rng)
            if x_levels is None:
                continue
            y_levels = _jagged_sequence(
                safe_h,
                v_segments + 1,
                min_step,
                rng,
                require_last_not_less=True,
            )
            if y_levels is None:
                continue
            points = [(y_levels[0], x_levels[0])]
            xi = 0
            yi = 0
            for s in range(segments):
                if s % 2 == 0:
                    xi += 1
                else:
                    yi += 1
                points.append((y_levels[yi], x_levels[xi]))
            return points

        v_segments = (segments + 1) // 2
        h_segments = segments // 2
        y_levels = _spaced_sorted_positions(safe_h, v_segments + 1, min_step, rng)
        if y_levels is None:
            continue
        x_levels = _jagged_sequence(safe_w, h_segments + 1, min_step, rng)
        if x_levels is None:
            continue
        points = [(y_levels[0], x_levels[0])]
        xi = 0
        yi = 0
        for s in range(segments):
            if s % 2 == 0:
                yi += 1
            else:
                xi += 1
            points.append((y_levels[yi], x_levels[xi]))
        return points
    return None


def _draw_path(grid: np.ndarray, points: list[tuple[int, int]]) -> None:
    for (r0, c0), (r1, c1) in zip(points, points[1:], strict=False):
        if r0 == r1:
            c_start, c_end = sorted((c0, c1))
            grid[r0, c_start : c_end + 1] = AZURE
        else:
            r_start, r_end = sorted((r0, r1))
            grid[r_start : r_end + 1, c0] = AZURE
    sr, sc = points[0]
    er, ec = points[-1]
    grid[sr, sc] = YELLOW
    grid[er, ec] = YELLOW


def _place_canvas_and_stamp(
    inner_h: int,
    inner_w: int,
    stamp: np.ndarray,
    side: str,
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int]]:
    canvas_h = inner_h + 2
    canvas_w = inner_w + 2
    stamp_h, stamp_w = stamp.shape
    gap = rng.randint(1, 3)
    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)

    if side in {"top", "bottom"}:
        extra_w = rng.randint(0, 5)
        grid_h = top_margin + bottom_margin + canvas_h + stamp_h + gap
        grid_w = left_margin + right_margin + max(canvas_w, stamp_w) + extra_w
    else:
        extra_h = rng.randint(0, 5)
        grid_h = top_margin + bottom_margin + max(canvas_h, stamp_h) + extra_h
        grid_w = left_margin + right_margin + canvas_w + stamp_w + gap

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    if side == "top":
        canvas_r0 = top_margin + stamp_h + gap
        stamp_r0 = top_margin
        canvas_c0 = rng.randint(left_margin, grid_w - right_margin - canvas_w)
        stamp_c0 = rng.randint(left_margin, grid_w - right_margin - stamp_w)
    elif side == "bottom":
        canvas_r0 = top_margin
        stamp_r0 = canvas_r0 + canvas_h + gap
        canvas_c0 = rng.randint(left_margin, grid_w - right_margin - canvas_w)
        stamp_c0 = rng.randint(left_margin, grid_w - right_margin - stamp_w)
    elif side == "left":
        canvas_c0 = left_margin + stamp_w + gap
        stamp_c0 = left_margin
        canvas_r0 = rng.randint(top_margin, grid_h - bottom_margin - canvas_h)
        stamp_r0 = rng.randint(top_margin, grid_h - bottom_margin - stamp_h)
    else:
        canvas_c0 = left_margin
        stamp_c0 = canvas_c0 + canvas_w + gap
        canvas_r0 = rng.randint(top_margin, grid_h - bottom_margin - canvas_h)
        stamp_r0 = rng.randint(top_margin, grid_h - bottom_margin - stamp_h)

    canvas_r1 = canvas_r0 + canvas_h - 1
    canvas_c1 = canvas_c0 + canvas_w - 1
    grid[canvas_r0, canvas_c0 : canvas_c1 + 1] = GREY
    grid[canvas_r1, canvas_c0 : canvas_c1 + 1] = GREY
    grid[canvas_r0 : canvas_r1 + 1, canvas_c0] = GREY
    grid[canvas_r0 : canvas_r1 + 1, canvas_c1] = GREY
    _paste(grid, stamp, stamp_r0, stamp_c0)

    return grid, (canvas_r0 + 1, canvas_c0 + 1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stamp = _make_stamp(rng)
    max_dim = max(stamp.shape)
    inner_h = rng.randint(max_dim + 4, max_dim + 10)
    inner_w = rng.randint(max_dim + 4, max_dim + 10)
    safe_h = inner_h - max_dim + 1
    safe_w = inner_w - max_dim + 1
    if safe_h < 2 and safe_w < 2:
        return None

    turns = rng.randint(0, 5)
    start_axis = rng.choice(["H", "V"])
    points = _generate_polyline(safe_h, safe_w, turns, start_axis, rng)
    if points is None:
        return None

    side = rng.choice(["top", "bottom", "left", "right"])
    grid, inner_origin = _place_canvas_and_stamp(inner_h, inner_w, stamp, side, rng)
    offset_r, offset_c = inner_origin
    shifted_points = [(offset_r + r, offset_c + c) for r, c in points]
    _draw_path(grid, shifted_points)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.count_nonzero(grid == YELLOW) != 2:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
