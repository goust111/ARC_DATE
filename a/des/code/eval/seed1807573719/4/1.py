import random
from typing import Dict, List, Tuple

import numpy as np


Grid = np.ndarray
Point = Tuple[int, int]


_GEN_BG = 0
_GEN_COLORS = tuple(range(1, 10))
_SIDES = ("up", "down", "left", "right")
_OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
_OBJECT_COUNT_WEIGHTS = (4, 3, 2, 1)
_GRID_SIDE_BOUNDS = {
    1: (12, 20),
    2: (16, 24),
    3: (20, 28),
    4: (24, 30),
}
_RECT_H_RANGE = (4, 8)
_RECT_W_RANGE = (4, 9)
_GEN_RETRIES = 500
_PLACE_RETRIES = 500
_TRUNCATE_SAMPLE_PROB = 0.5


def get_bg(grid: Grid) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> List[Point]:
    pts: List[Point] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return list(dict.fromkeys(pts))


def _bbox_cells(r0: int, c0: int, r1: int, c1: int) -> List[Point]:
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _path_cells(seed: Point, direction: Point, length: int, h: int, w: int) -> List[Point]:
    dr, dc = direction
    r, c = seed
    pts: List[Point] = []
    for _ in range(length):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        pts.append((r, c))
    return pts


def _mark_blocked(blocked: np.ndarray, cells: List[Point]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _frame_direction(r0: int, c0: int, r1: int, c1: int, gap: Point) -> Point | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return (-1, 0)
    if gr == r1 and c0 < gc < c1:
        return (1, 0)
    if gc == c0 and r0 < gr < r1:
        return (0, -1)
    if gc == c1 and r0 < gr < r1:
        return (0, 1)
    return None


def _extract_frames(grid: Grid) -> List[Dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    frames: List[Dict[str, object]] = []
    used_gap_colors: set[int] = set()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        pts = np.argwhere(g == color)
        if pts.shape[0] < 7:
            continue

        r0 = int(pts[:, 0].min())
        c0 = int(pts[:, 1].min())
        r1 = int(pts[:, 0].max())
        c1 = int(pts[:, 1].max())
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue

        border = _border_cells(r0, c0, r1, c1)
        border_set = set(border)
        if len(border) - 1 != int(pts.shape[0]):
            continue
        if any((int(r), int(c)) not in border_set for r, c in pts):
            continue

        gap_candidates = [(r, c) for (r, c) in border if int(g[r, c]) != color]
        if len(gap_candidates) != 1:
            continue
        gap = gap_candidates[0]
        gap_color = int(g[gap])
        if gap_color in (bg, color) or gap_color in used_gap_colors:
            continue

        direction = _frame_direction(r0, c0, r1, c1, gap)
        if direction is None:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size < 2:
            continue
        if np.any((interior != bg) & (interior != gap_color)):
            continue

        seed_rel = np.argwhere(interior == gap_color)
        if seed_rel.shape != (1, 2):
            continue
        seed = (r0 + 1 + int(seed_rel[0, 0]), c0 + 1 + int(seed_rel[0, 1]))

        if int(np.count_nonzero(g == gap_color)) != 2:
            continue

        dr, dc = direction
        if dr != 0 and seed[1] != gap[1]:
            continue
        if dc != 0 and seed[0] != gap[0]:
            continue

        rr, cc = seed
        clear = True
        while True:
            rr += dr
            cc += dc
            if (rr, cc) == gap:
                break
            if not (0 <= rr < h and 0 <= cc < w) or int(g[rr, cc]) != bg:
                clear = False
                break
        if not clear:
            continue

        length = int(np.count_nonzero(interior == bg))
        if length <= 0:
            continue

        frames.append(
            {
                "frame_color": color,
                "gap_color": gap_color,
                "bbox": (r0, c0, r1, c1),
                "seed": seed,
                "gap": gap,
                "direction": direction,
                "length": length,
            }
        )
        used_gap_colors.add(gap_color)

    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed to the border gap of the same color, then grow a border-colored
    path from the seed through that gap. The path length equals the count of
    background pixels inside the corresponding rectangle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    frames = _extract_frames(g)
    assert frames, "expected at least one valid frame"

    for frame in frames:
        seed = frame["seed"]
        direction = frame["direction"]
        length = int(frame["length"])
        frame_color = int(frame["frame_color"])

        for r, c in _path_cells(seed, direction, length, h, w):
            out[r, c] = frame_color

    return out


def _sample_object_count(rng: random.Random) -> int:
    return rng.choices(_OBJECT_COUNT_CHOICES, weights=_OBJECT_COUNT_WEIGHTS, k=1)[0]


def _origin_bounds(
    axis_len: int,
    outer_len: int,
    seed_offset: int,
    length: int,
    toward_low_end: bool,
    need_truncation: bool,
) -> Tuple[int, int]:
    lo = 1
    hi = axis_len - outer_len - 1
    if toward_low_end:
        if need_truncation:
            hi = min(hi, length - seed_offset - 1)
        else:
            lo = max(lo, length - seed_offset)
    else:
        if need_truncation:
            lo = max(lo, axis_len - seed_offset - length)
        else:
            hi = min(hi, axis_len - 1 - seed_offset - length)
    return lo, hi


def _place_object(
    rng: random.Random,
    grid: Grid,
    blocked: np.ndarray,
    frame_color: int,
    pair_color: int,
    need_truncation: bool,
) -> Dict[str, object] | None:
    h, w = grid.shape
    max_outer_h = min(_RECT_H_RANGE[1], h - 2)
    max_outer_w = min(_RECT_W_RANGE[1], w - 2)
    if max_outer_h < _RECT_H_RANGE[0] or max_outer_w < _RECT_W_RANGE[0]:
        return None

    for _ in range(_PLACE_RETRIES):
        outer_h = rng.randint(_RECT_H_RANGE[0], max_outer_h)
        outer_w = rng.randint(_RECT_W_RANGE[0], max_outer_w)
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        length = inner_h * inner_w - 1
        if length <= 0:
            continue

        side = rng.choice(_SIDES)
        if side in ("up", "down"):
            gap_offset = rng.randint(1, outer_w - 2)
            seed_offset = rng.randint(1, outer_h - 2)
            r_lo, r_hi = _origin_bounds(h, outer_h, seed_offset, length, side == "up", need_truncation)
            c_lo, c_hi = 1, w - outer_w - 1
            if r_lo > r_hi or c_lo > c_hi:
                continue
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            gap = (r0, c0 + gap_offset) if side == "up" else (r1, c0 + gap_offset)
            seed = (r0 + seed_offset, c0 + gap_offset)
            direction = (-1, 0) if side == "up" else (1, 0)
        else:
            gap_offset = rng.randint(1, outer_h - 2)
            seed_offset = rng.randint(1, outer_w - 2)
            r_lo, r_hi = 1, h - outer_h - 1
            c_lo, c_hi = _origin_bounds(w, outer_w, seed_offset, length, side == "left", need_truncation)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            gap = (r0 + gap_offset, c0) if side == "left" else (r0 + gap_offset, c1)
            seed = (r0 + gap_offset, c0 + seed_offset)
            direction = (0, -1) if side == "left" else (0, 1)

        path = _path_cells(seed, direction, length, h, w)
        is_truncated = len(path) < length
        if is_truncated != need_truncation:
            continue

        bbox_cells = _bbox_cells(r0, c0, r1, c1)
        bbox_set = set(bbox_cells)
        reserved = bbox_cells + [pt for pt in path if pt not in bbox_set]
        if any(blocked[r, c] for r, c in reserved):
            continue

        border = _border_cells(r0, c0, r1, c1)
        for r, c in border:
            if (r, c) != gap:
                grid[r, c] = frame_color
        grid[gap] = pair_color
        grid[seed] = pair_color
        _mark_blocked(blocked, reserved)

        return {
            "frame_color": frame_color,
            "pair_color": pair_color,
            "bbox": (r0, c0, r1, c1),
            "seed": seed,
            "gap": gap,
            "direction": direction,
            "length": length,
            "is_truncated": is_truncated,
        }

    return None


def _gen_attempt(rng: random.Random) -> Grid | None:
    num_objects = _sample_object_count(rng)
    min_side, max_side = _GRID_SIDE_BOUNDS[num_objects]
    h = rng.randint(min_side, max_side)
    w = rng.randint(min_side, max_side)
    grid = np.full((h, w), _GEN_BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = rng.sample(_GEN_COLORS, 2 * num_objects)
    frame_colors = colors[:num_objects]
    pair_colors = colors[num_objects:]

    want_truncation = rng.random() < _TRUNCATE_SAMPLE_PROB
    trunc_flags = [False] * num_objects
    if want_truncation:
        for idx in range(num_objects):
            trunc_flags[idx] = rng.random() < 0.5
        if not any(trunc_flags):
            trunc_flags[rng.randrange(num_objects)] = True

    order = list(range(num_objects))
    order.sort(key=lambda idx: (not trunc_flags[idx], rng.random()))

    placed: List[Dict[str, object]] = []
    for idx in order:
        obj = _place_object(
            rng,
            grid,
            blocked,
            frame_colors[idx],
            pair_colors[idx],
            trunc_flags[idx],
        )
        if obj is None:
            return None
        placed.append(obj)

    if get_bg(grid) != _GEN_BG:
        return None

    frames = _extract_frames(grid)
    if len(frames) != num_objects:
        return None
    if {int(frame["frame_color"]) for frame in frames} != set(frame_colors):
        return None
    if {int(frame["gap_color"]) for frame in frames} != set(pair_colors):
        return None
    if want_truncation and not any(len(_path_cells(frame["seed"], frame["direction"], int(frame["length"]), h, w)) < int(frame["length"]) for frame in frames):
        return None
    if not want_truncation and any(len(_path_cells(frame["seed"], frame["direction"], int(frame["length"]), h, w)) < int(frame["length"]) for frame in frames):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
