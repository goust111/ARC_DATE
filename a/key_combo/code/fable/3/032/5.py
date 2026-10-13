from __future__ import annotations

from collections import defaultdict, deque
import bisect
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

UP, RIGHT, DOWN, LEFT = range(4)
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

GEN_ATTEMPTS = 500
OUTER_HEIGHT_RANGE = (10, 18)
OUTER_WIDTH_RANGE = (10, 18)
TOP_COUNT_CHOICES = [0, 1, 1, 2, 2, 3]
BOTTOM_COUNT_CHOICES = [0, 0, 1, 1, 2]
MIN_NOTCH_SPAN = 2
MAX_NOTCH_SPAN = 5
MIN_NOTCH_DEPTH = 2
MAX_NOTCH_DEPTH = 6
NOTCH_EDGE_MARGIN = 2
NOTCH_GAP = 2
PAD_LIMIT = 4


def _turn(direction: int, delta: int) -> int:
    return (int(direction) + int(delta)) % 4


def _opposite(direction: int) -> int:
    return (int(direction) + 2) % 4


def _segment_direction(a: tuple[int, int], b: tuple[int, int]) -> int:
    ar, ac = a
    br, bc = b
    if ar == br:
        return RIGHT if bc > ac else LEFT
    return DOWN if br > ar else UP


def _extract_corners(grid: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    coords: list[tuple[int, int]] = []
    colors: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(g != BLACK):
        color = int(g[r, c])
        assert color in (YELLOW, RED), "input must contain only yellow/red corner pixels"
        coord = (int(r), int(c))
        coords.append(coord)
        colors[coord] = color

    assert len(coords) >= 4 and len(coords) % 2 == 0, "expected an even number of corners"
    return sorted(coords), colors


def _build_line_indexes(
    coords: list[tuple[int, int]],
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[tuple[int, int], int]]:
    row_cols: dict[int, list[int]] = defaultdict(list)
    col_rows: dict[int, list[int]] = defaultdict(list)
    coord_to_idx: dict[tuple[int, int], int] = {}
    for idx, (r, c) in enumerate(coords):
        row_cols[r].append(c)
        col_rows[c].append(r)
        coord_to_idx[(r, c)] = idx
    for cols in row_cols.values():
        cols.sort()
    for rows in col_rows.values():
        rows.sort()
    return row_cols, col_rows, coord_to_idx


def _nearest_corner(
    coords: list[tuple[int, int]],
    row_cols: dict[int, list[int]],
    col_rows: dict[int, list[int]],
    coord_to_idx: dict[tuple[int, int], int],
    idx: int,
    direction: int,
) -> int | None:
    r, c = coords[idx]
    if direction == RIGHT:
        cols = row_cols[r]
        pos = bisect.bisect_right(cols, c)
        if pos >= len(cols):
            return None
        return coord_to_idx[(r, cols[pos])]
    if direction == LEFT:
        cols = row_cols[r]
        pos = bisect.bisect_left(cols, c) - 1
        if pos < 0:
            return None
        return coord_to_idx[(r, cols[pos])]
    if direction == DOWN:
        rows = col_rows[c]
        pos = bisect.bisect_right(rows, r)
        if pos >= len(rows):
            return None
        return coord_to_idx[(rows[pos], c)]
    rows = col_rows[c]
    pos = bisect.bisect_left(rows, r) - 1
    if pos < 0:
        return None
    return coord_to_idx[(rows[pos], c)]


def _trace_candidate(
    coords: list[tuple[int, int]],
    colors: dict[tuple[int, int], int],
    row_cols: dict[int, list[int]],
    col_rows: dict[int, list[int]],
    coord_to_idx: dict[tuple[int, int], int],
    start_idx: int,
    incoming_dir: int,
    right_turn_color: int,
) -> list[tuple[int, int]] | None:
    n = len(coords)
    seen: set[int] = set()
    segments: list[tuple[int, int]] = []
    idx = int(start_idx)
    cur_in = int(incoming_dir)

    for step in range(n):
        if idx in seen:
            return None
        seen.add(idx)

        turn_delta = 1 if colors[coords[idx]] == right_turn_color else -1
        out_dir = _turn(cur_in, turn_delta)
        nxt = _nearest_corner(coords, row_cols, col_rows, coord_to_idx, idx, out_dir)
        if nxt is None:
            return None

        next_in = _opposite(out_dir)
        segments.append((idx, nxt))

        if nxt in seen:
            if nxt == start_idx and step == n - 1 and next_in == incoming_dir:
                return segments
            return None

        idx = nxt
        cur_in = next_in

    return None


def _render_segments(
    shape: tuple[int, int],
    coords: list[tuple[int, int]],
    colors: dict[tuple[int, int], int],
    segments: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for a_idx, b_idx in segments:
        ar, ac = coords[a_idx]
        br, bc = coords[b_idx]
        if ar == br:
            c0, c1 = sorted((ac, bc))
            out[ar, c0 : c1 + 1] = GREY
        else:
            r0, r1 = sorted((ar, br))
            out[r0 : r1 + 1, ac] = GREY
    for coord, color in colors.items():
        out[coord] = int(color)
    return out


def _occupied_neighbors(mask: np.ndarray, r: int, c: int) -> tuple[bool, bool, bool, bool]:
    h, w = mask.shape
    up = r > 0 and bool(mask[r - 1, c])
    right = c + 1 < w and bool(mask[r, c + 1])
    down = r + 1 < h and bool(mask[r + 1, c])
    left = c > 0 and bool(mask[r, c - 1])
    return up, right, down, left


def _is_valid_loop_output(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    colors: dict[tuple[int, int], int],
) -> bool:
    mask = out != BLACK
    occupied = np.argwhere(mask)
    if occupied.size == 0:
        return False

    start = tuple(map(int, occupied[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and mask[nr, nc]:
                coord = (nr, nc)
                if coord not in seen:
                    seen.add(coord)
                    q.append(coord)
    if len(seen) != int(mask.sum()):
        return False

    corner_set = set(coords)
    for r, c in map(tuple, occupied):
        up, right, down, left = _occupied_neighbors(mask, int(r), int(c))
        degree = int(up) + int(right) + int(down) + int(left)
        if degree != 2:
            return False

        is_bend = (up or down) and (left or right)
        coord = (int(r), int(c))
        color = int(out[r, c])
        if coord in corner_set:
            if color != colors[coord] or not is_bend:
                return False
        else:
            if color != GREY or is_bend:
                return False

    return True


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    g = np.asarray(grid, dtype=int)
    return (int(g.shape[0]), int(g.shape[1])), g.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden one-cell-thick orthogonal loop from its convex and concave bend pixels.

    Input structure:
    - The grid has a black background and only isolated yellow/red foreground pixels.
    - Every foreground pixel is a bend of a hidden simple closed orthogonal loop.
    - Yellow marks convex corners and red marks concave corners; all straight run pixels are missing.

    Rule:
    - Follow the unique loop that connects corners by straight horizontal/vertical segments between
      same-row or same-column corners.
    - Along some traversal direction, all corners of one color are right turns and the other color
      are left turns; this turn pattern resolves which visible corner is next on each ray.
    - Draw every recovered straight run in grey while preserving the original yellow/red corners.

    Key insight:
    - The corner colors are not decoration: they encode the turn sign at each bend, so once an
      incoming direction is hypothesized, the outgoing direction and therefore the next corner are
      forced; only one traced cycle uses every corner exactly once without branching or crossings.
    """
    g = np.asarray(grid, dtype=int)
    coords, colors = _extract_corners(g)
    row_cols, col_rows, coord_to_idx = _build_line_indexes(coords)

    solutions: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for right_turn_color in (YELLOW, RED):
        for start_idx in range(len(coords)):
            for incoming_dir in (UP, RIGHT, DOWN, LEFT):
                segments = _trace_candidate(
                    coords,
                    colors,
                    row_cols,
                    col_rows,
                    coord_to_idx,
                    start_idx,
                    incoming_dir,
                    right_turn_color,
                )
                if segments is None:
                    continue
                out = _render_segments(g.shape, coords, colors, segments)
                if _is_valid_loop_output(out, coords, colors):
                    solutions[_grid_signature(out)] = out

    assert len(solutions) == 1, "expected exactly one valid loop reconstruction"
    return next(iter(solutions.values()))


def _sample_notch_spans(width: int, count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []

    max_span = min(MAX_NOTCH_SPAN, width - 2 * NOTCH_EDGE_MARGIN - 1)
    if max_span < MIN_NOTCH_SPAN:
        return None

    spans: list[tuple[int, int]] = []
    for _ in range(120):
        spans.clear()
        while len(spans) < count:
            span = random.randint(MIN_NOTCH_SPAN, max_span)
            c0 = random.randint(NOTCH_EDGE_MARGIN, width - 1 - NOTCH_EDGE_MARGIN - span)
            c1 = c0 + span
            if any(not (c1 + NOTCH_GAP < s0 or c0 > s1 + NOTCH_GAP) for s0, s1 in spans):
                break
            spans.append((c0, c1))
        if len(spans) == count:
            return sorted(spans)
    return None


def _build_base_corners(
    height: int,
    width: int,
    top_notches: list[tuple[int, int, int]],
    bottom_notches: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    corners: list[tuple[int, int]] = [(0, 0)]

    for c0, c1, depth in top_notches:
        corners.extend(
            [
                (0, c0),
                (depth, c0),
                (depth, c1),
                (0, c1),
            ]
        )

    corners.extend([(0, width - 1), (height - 1, width - 1)])

    for c0, c1, depth in reversed(bottom_notches):
        inner_row = height - 1 - depth
        corners.extend(
            [
                (height - 1, c1),
                (inner_row, c1),
                (inner_row, c0),
                (height - 1, c0),
            ]
        )

    corners.append((height - 1, 0))
    return corners


def _corner_colors_from_cycle(corners: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    n = len(corners)
    for idx, cur in enumerate(corners):
        prev = corners[idx - 1]
        nxt = corners[(idx + 1) % n]
        d_in = _segment_direction(prev, cur)
        d_out = _segment_direction(cur, nxt)
        delta = (d_out - d_in) % 4
        if delta == 1:
            colors[cur] = YELLOW
        elif delta == 3:
            colors[cur] = RED
        else:
            raise AssertionError("cycle must turn at every listed corner")
    return colors


def _render_cycle(
    height: int,
    width: int,
    corners: list[tuple[int, int]],
    colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    n = len(corners)
    for idx in range(n):
        ar, ac = corners[idx]
        br, bc = corners[(idx + 1) % n]
        if ar == br:
            c0, c1 = sorted((ac, bc))
            out[ar, c0 : c1 + 1] = GREY
        else:
            r0, r1 = sorted((ar, br))
            out[r0 : r1 + 1, ac] = GREY
    for coord, color in colors.items():
        out[coord] = int(color)
    return out


def _strip_segments(full_loop: np.ndarray) -> np.ndarray:
    inp = np.zeros_like(full_loop)
    inp[full_loop == YELLOW] = YELLOW
    inp[full_loop == RED] = RED
    return inp


def _apply_shared_symmetry(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair = np.stack([inp, out], axis=0)
    pair = np.rot90(pair, random.randint(0, 3), axes=(1, 2))
    if random.random() < 0.5:
        pair = np.flip(pair, axis=2)
    if random.random() < 0.5:
        pair = np.flip(pair, axis=1)
    return np.ascontiguousarray(pair[0]), np.ascontiguousarray(pair[1])


def _apply_shared_padding(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, inp.shape)
    assert inp.shape == out.shape
    max_top = min(PAD_LIMIT, 30 - h)
    top = random.randint(0, max_top)
    max_bottom = min(PAD_LIMIT, 30 - h - top)
    bottom = random.randint(0, max_bottom)
    max_left = min(PAD_LIMIT, 30 - w)
    left = random.randint(0, max_left)
    max_right = min(PAD_LIMIT, 30 - w - left)
    right = random.randint(0, max_right)

    shape = (h + top + bottom, w + left + right)
    inp_out = np.zeros(shape, dtype=int)
    full_out = np.zeros(shape, dtype=int)
    inp_out[top : top + h, left : left + w] = inp
    full_out[top : top + h, left : left + w] = out
    return inp_out, full_out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*OUTER_HEIGHT_RANGE)
    width = random.randint(*OUTER_WIDTH_RANGE)

    top_count = random.choice(TOP_COUNT_CHOICES)
    bottom_count = random.choice(BOTTOM_COUNT_CHOICES)
    if top_count + bottom_count == 0 and random.random() < 0.9:
        top_count = 1

    top_spans = _sample_notch_spans(width, top_count)
    bottom_spans = _sample_notch_spans(width, bottom_count)
    if top_spans is None or bottom_spans is None:
        return None

    max_depth = min(MAX_NOTCH_DEPTH, height - 4)
    if max_depth < MIN_NOTCH_DEPTH:
        return None

    top_notches = [(c0, c1, random.randint(MIN_NOTCH_DEPTH, max_depth)) for c0, c1 in top_spans]
    bottom_notches = [(c0, c1, random.randint(MIN_NOTCH_DEPTH, max_depth)) for c0, c1 in bottom_spans]
    corners = _build_base_corners(height, width, top_notches, bottom_notches)

    if len(set(corners)) != len(corners):
        return None

    colors = _corner_colors_from_cycle(corners)
    full_loop = _render_cycle(height, width, corners, colors)
    if not _is_valid_loop_output(full_loop, sorted(colors), colors):
        return None

    inp = _strip_segments(full_loop)
    inp, out = _apply_shared_symmetry(inp, full_loop)
    inp, out = _apply_shared_padding(inp, out)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        generated = _gen_attempt()
        if generated is None:
            continue
        inp, expected = generated
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid corner-only orthogonal loop instance")
