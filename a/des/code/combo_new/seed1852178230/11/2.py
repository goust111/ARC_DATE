from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG_COLOR = 0
SEPARATOR_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
BOX_SIZE = 5
BOX_GAP_OPTIONS = (1, 2)
GEN_DIRECTION_OPTIONS = ("down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400

ARROW_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "up": ((2, 2), (1, 2), (0, 2), (1, 1), (1, 3)),
    "down": ((2, 2), (3, 2), (4, 2), (3, 1), (3, 3)),
    "left": ((2, 2), (2, 1), (2, 0), (1, 1), (3, 1)),
    "right": ((2, 2), (2, 3), (2, 4), (1, 3), (3, 3)),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _segment_midpoint(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        return int(r0), int(c0 + (c1 - c0) // 2)
    if c0 == c1:
        return int(r0 + (r1 - r0) // 2), int(c0)
    raise AssertionError(f"component is not a 1-cell-wide segment: bbox={_bbox(coords)}")


def _is_segment(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        return int(coords.shape[0]) == int(c1 - c0 + 1)
    if c0 == c1:
        return int(coords.shape[0]) == int(r1 - r0 + 1)
    return False


def _segment_order_key(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = _bbox(coords)
    mr, mc = _segment_midpoint(coords)
    return int(r0), int(c0), int(mr), int(mc)


def _pair_direction(segments: Iterable[np.ndarray]) -> str:
    first, second = sorted(list(segments), key=_segment_order_key)
    (r1, c1) = _segment_midpoint(first)
    (r2, c2) = _segment_midpoint(second)
    dr = int(r2 - r1)
    dc = int(c2 - c1)
    if dr == 0 and dc == 0:
        raise AssertionError("paired segments cannot share the same midpoint")
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    if dr != 0:
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _find_separator_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != bg and bool(np.all(col == col[0])):
            candidates.append((int(c), int(col[0])))
    assert len(candidates) == 1, f"expected one solid separator column, got {candidates}"
    return candidates[0]


def _is_frame(grid: np.ndarray, coords: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (BOX_SIZE, BOX_SIZE):
        return False
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(
        np.all(box[0, :] == color)
        and np.all(box[-1, :] == color)
        and np.all(box[:, 0] == color)
        and np.all(box[:, -1] == color)
    )


def _draw_arrow(grid: np.ndarray, top: int, left: int, color: int, direction: str) -> None:
    for dr, dc in ARROW_OFFSETS[direction]:
        grid[int(top + dr), int(left + dc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored line segments and encode their displacement as arrows in matching framed boxes.

    Input structure:
    - The background is black, and a single solid gray column separates the grid into left and right regions.
    - The left region contains colored 1-cell-wide horizontal or vertical line segments; each color appears in exactly two disconnected segments.
    - The right region contains a vertical stack of 5x5 square frames whose border colors match the segment colors.

    Rule:
    - For each color, identify the two left-side segments and order them by vertical position using the segment whose topmost row is higher as the first one.
    - Compute the midpoint of each segment, using the upper/left middle cell for even lengths, then convert the vector from the first midpoint to the second midpoint into the nearest cardinal direction.
    - Draw a same-colored arrow in the corresponding right-side 5x5 frame: start at the frame center, extend a 2-step shaft in that direction, and add the two diagonal tip pixels at the shaft end.

    Key insight:
    - The left-side colored paths are not meant to be connected; their relative displacement encodes a direction, and the right-side colored frame simply tells where to render that decoded arrow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    sep_c, sep_color = _find_separator_column(g, bg)

    colors = sorted(int(v) for v in np.unique(g) if int(v) not in {bg, sep_color})
    frames: dict[int, tuple[int, int]] = {}
    segments_by_color: dict[int, list[np.ndarray]] = {}

    for color in colors:
        comps = _components(g == int(color))
        left_segments = []
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            if c1 < sep_c:
                assert _is_segment(coords), f"left object is not a valid segment: color={color}, bbox={_bbox(coords)}"
                left_segments.append(coords)
            elif c0 > sep_c and _is_frame(g, coords, int(color)):
                frames[int(color)] = (int(r0), int(c0))
        if left_segments:
            segments_by_color[int(color)] = left_segments

    assert frames, "no right-side frames found"

    out = g.copy()
    for color, (top, left) in frames.items():
        segments = segments_by_color.get(int(color))
        assert segments is not None and len(segments) == 2, f"color {color} must have exactly two segments"
        direction = _pair_direction(segments)
        _draw_arrow(out, int(top), int(left), int(color), direction)

    return out


def _make_segment(orientation: str, start_r: int, start_c: int, length: int) -> np.ndarray:
    if orientation == "h":
        return np.array([(int(start_r), int(start_c + i)) for i in range(length)], dtype=int)
    return np.array([(int(start_r + i), int(start_c)) for i in range(length)], dtype=int)


def _sample_segment(
    rng: random.Random,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
) -> np.ndarray | None:
    options = []
    if col_hi - col_lo + 1 >= 2:
        options.append("h")
    if row_hi - row_lo + 1 >= 2:
        options.append("v")
    if not options:
        return None

    orientation = rng.choice(options)
    if orientation == "h":
        length = rng.randint(2, min(5, col_hi - col_lo + 1))
        row = rng.randint(row_lo, row_hi)
        col = rng.randint(col_lo, col_hi - length + 1)
    else:
        length = rng.randint(2, min(5, row_hi - row_lo + 1))
        row = rng.randint(row_lo, row_hi - length + 1)
        col = rng.randint(col_lo, col_hi)
    return _make_segment(orientation, int(row), int(col), int(length))


def _can_place_segment(segment: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = map(int, blocked.shape)
    r0, c0, r1, c1 = _bbox(segment)
    rr0 = max(0, int(r0 - 1))
    rr1 = min(h - 1, int(r1 + 1))
    cc0 = max(0, int(c0 - 1))
    cc1 = min(w - 1, int(c1 + 1))
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_segment(segment: np.ndarray, blocked: np.ndarray) -> None:
    h, w = map(int, blocked.shape)
    r0, c0, r1, c1 = _bbox(segment)
    rr0 = max(0, int(r0 - 1))
    rr1 = min(h - 1, int(r1 + 1))
    cc0 = max(0, int(c0 - 1))
    cc1 = min(w - 1, int(c1 + 1))
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _choose_directions(rng: random.Random, count: int) -> list[str]:
    if count == 2:
        return rng.sample(list(GEN_DIRECTION_OPTIONS), 2)
    base = list(GEN_DIRECTION_OPTIONS)
    rng.shuffle(base)
    while len(base) < count:
        base.append(rng.choice(GEN_DIRECTION_OPTIONS))
    return base[:count]


def _paint_segment(grid: np.ndarray, segment: np.ndarray, color: int) -> None:
    grid[segment[:, 0], segment[:, 1]] = int(color)


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(2, 4)
    box_gap = rng.choice(BOX_GAP_OPTIONS)
    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)

    height = top_margin + color_count * BOX_SIZE + (color_count - 1) * box_gap + bottom_margin
    if height > 30:
        return None

    sep_gap = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    min_left_width = 10
    max_left_width = 30 - (1 + sep_gap + BOX_SIZE + right_margin)
    if max_left_width < min_left_width:
        return None
    left_width = rng.randint(min_left_width, max_left_width)
    width = left_width + 1 + sep_gap + BOX_SIZE + right_margin
    if width > 30:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    sep_c = left_width
    grid[:, sep_c] = SEPARATOR_COLOR

    colors = rng.sample(PALETTE, color_count)
    directions = _choose_directions(rng, color_count)
    blocked = np.zeros((height, left_width), dtype=bool)

    segment_pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    box_rows: list[tuple[int, int]] = []
    for idx, color in enumerate(colors):
        box_top = top_margin + idx * (BOX_SIZE + box_gap)
        box_rows.append((int(color), int(box_top)))
        row_lo = int(box_top)
        row_hi = int(box_top + BOX_SIZE - 1)
        target_direction = directions[idx]

        pair = None
        for _ in range(600):
            temp_blocked = blocked.copy()
            segs: list[np.ndarray] = []
            for _ in range(2):
                placed = None
                for _ in range(200):
                    candidate = _sample_segment(rng, row_lo, row_hi, 1, left_width - 2)
                    if candidate is not None and _can_place_segment(candidate, temp_blocked):
                        _mark_segment(candidate, temp_blocked)
                        placed = candidate
                        break
                if placed is None:
                    segs = []
                    break
                segs.append(placed)
            if len(segs) != 2:
                continue
            top_rows = sorted(_bbox(seg)[0] for seg in segs)
            if top_rows[0] == top_rows[1]:
                continue
            first, second = sorted(segs, key=_segment_order_key)
            dr = _segment_midpoint(second)[0] - _segment_midpoint(first)[0]
            dc = _segment_midpoint(second)[1] - _segment_midpoint(first)[1]
            if abs(dr) == abs(dc):
                continue
            if _pair_direction(segs) != target_direction:
                continue
            pair = (segs[0], segs[1])
            blocked = temp_blocked
            break

        if pair is None:
            return None

        segment_pairs[int(color)] = pair
        _paint_segment(grid, pair[0], int(color))
        _paint_segment(grid, pair[1], int(color))

    box_left = sep_c + 1 + sep_gap
    ordered_colors = sorted(segment_pairs, key=lambda color: min(_bbox(seg)[0] for seg in segment_pairs[color]))
    expected_order = [color for color, _row in box_rows]
    if ordered_colors != expected_order:
        return None

    for color, box_top in box_rows:
        grid[box_top : box_top + BOX_SIZE, box_left : box_left + BOX_SIZE] = BG_COLOR
        grid[box_top, box_left : box_left + BOX_SIZE] = color
        grid[box_top + BOX_SIZE - 1, box_left : box_left + BOX_SIZE] = color
        grid[box_top : box_top + BOX_SIZE, box_left] = color
        grid[box_top : box_top + BOX_SIZE, box_left + BOX_SIZE - 1] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not bool(np.any(out != grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
