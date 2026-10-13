from __future__ import annotations

import random

import numpy as np


BG = 5
START = 3
HEAD = 8
LINK = 0

SEGMENT_COLORS = (1, 2, 6, 7, 9)
LENGTH_VALUES = (2, 3, 4, 5, 6)
ALL_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
SEGMENT_COUNT_RANGE = (3, 5)

PATH_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _row_major_first(coords: np.ndarray) -> tuple[int, int]:
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    r, c = coords[int(order[0])]
    return int(r), int(c)


def _extract_segments(grid: np.ndarray) -> list[dict]:
    head_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == HEAD)}
    segment_colors = [int(c) for c in np.unique(grid) if int(c) not in {BG, START, HEAD}]
    assert segment_colors, "expected at least one colored segment"

    used_heads: set[tuple[int, int]] = set()
    segments: list[dict] = []

    for color in segment_colors:
        coords = np.argwhere(grid == color)
        assert coords.shape[0] >= 2, "segments must have length >= 2"
        rows = coords[:, 0]
        cols = coords[:, 1]
        length = int(coords.shape[0])
        assert 2 <= length <= 6, "segment length must be 2-6"

        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if np.all(rows == rows[0]):
            r = int(rows[0])
            sorted_cols = np.sort(cols)
            c0, c1 = int(sorted_cols[0]), int(sorted_cols[-1])
            assert np.array_equal(sorted_cols, np.arange(c0, c1 + 1)), "horizontal segment must be contiguous"
            assert length == c1 - c0 + 1
            if (r, c0 - 1) in head_cells:
                candidates.append(((r, c0 - 1), (0, 1)))
            if (r, c1 + 1) in head_cells:
                candidates.append(((r, c1 + 1), (0, -1)))
        elif np.all(cols == cols[0]):
            c = int(cols[0])
            sorted_rows = np.sort(rows)
            r0, r1 = int(sorted_rows[0]), int(sorted_rows[-1])
            assert np.array_equal(sorted_rows, np.arange(r0, r1 + 1)), "vertical segment must be contiguous"
            assert length == r1 - r0 + 1
            if (r0 - 1, c) in head_cells:
                candidates.append(((r0 - 1, c), (1, 0)))
            if (r1 + 1, c) in head_cells:
                candidates.append(((r1 + 1, c), (-1, 0)))
        else:
            raise AssertionError("each segment must be horizontal or vertical")

        assert len(candidates) == 1, f"segment color {color} must have exactly one head marker"
        head_pos, direction = candidates[0]
        assert head_pos not in used_heads, "each azure head must belong to exactly one segment"
        used_heads.add(head_pos)

        first_row, first_col = _row_major_first(coords)
        segments.append(
            {
                "color": int(color),
                "length": length,
                "direction": direction,
                "first_row": first_row,
                "first_col": first_col,
            }
        )

    assert used_heads == head_cells, "all azure markers must be used"
    return sorted(segments, key=lambda seg: (seg["length"], seg["first_row"], seg["first_col"]))


def _draw_polyline(shape: tuple[int, int], start: tuple[int, int], segments: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[start] = START
    cursor = start

    for idx, seg in enumerate(segments):
        if idx == 0:
            anchor = start
        else:
            anchor = (cursor[0] + 1, cursor[1])
            assert _in_bounds(shape, anchor), "connector exits the canvas"
            out[anchor] = LINK

        dr, dc = seg["direction"]
        length = int(seg["length"])
        for step in range(1, length + 1):
            pos = (anchor[0] + dr * step, anchor[1] + dc * step)
            assert _in_bounds(shape, pos), "redrawn segment exits the canvas"
            out[pos] = int(seg["color"])
        cursor = (anchor[0] + dr * length, anchor[1] + dc * length)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the scattered colored segments by length, decode each segment's direction
    from its azure head marker, and redraw the segments as one continuous polyline
    starting at the green pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    starts = np.argwhere(g == START)
    assert starts.shape[0] == 1, "expected exactly one green start marker"
    start = tuple(map(int, starts[0]))

    ordered_segments = _extract_segments(g)
    return _draw_polyline(g.shape, start, ordered_segments)


def _sample_lengths(count: int) -> list[int]:
    if count <= len(LENGTH_VALUES) and random.random() < 0.45:
        lengths = random.sample(LENGTH_VALUES, k=count)
    else:
        while True:
            lengths = [random.choice(LENGTH_VALUES) for _ in range(count)]
            if len(set(lengths)) > 1 and any(lengths.count(v) > 1 for v in set(lengths)):
                break
    return sorted(lengths)


def _try_build_output(shape: tuple[int, int], base_segments: list[dict]) -> tuple[tuple[int, int], list[dict], np.ndarray] | None:
    H, W = shape

    for _ in range(PATH_ATTEMPTS):
        start = (random.randint(2, H - 4), random.randint(2, W - 4))
        cursor = start
        occupied = {start}
        built: list[dict] = []
        orientations: set[str] = set()
        directions_seen: set[tuple[int, int]] = set()
        ok = True

        for idx, base in enumerate(base_segments):
            anchor = start if idx == 0 else (cursor[0] + 1, cursor[1])
            if not _in_bounds(shape, anchor):
                ok = False
                break
            if idx > 0 and anchor in occupied:
                ok = False
                break

            options: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
            directions = list(ALL_DIRECTIONS)
            random.shuffle(directions)
            for direction in directions:
                dr, dc = direction
                cells = [(anchor[0] + dr * step, anchor[1] + dc * step) for step in range(1, base["length"] + 1)]
                if all(_in_bounds(shape, cell) and cell not in occupied for cell in cells):
                    options.append((direction, cells))

            if not options:
                ok = False
                break

            direction, cells = random.choice(options)
            seg = dict(base)
            seg["direction"] = direction
            built.append(seg)

            occupied.add(anchor)
            occupied.update(cells)
            cursor = cells[-1]
            directions_seen.add(direction)
            orientations.add("v" if direction[0] else "h")

        if ok and len(orientations) == 2 and len(directions_seen) >= 2:
            return start, built, _draw_polyline(shape, start, built)

    return None


def _segment_patch(color: int, length: int, direction: tuple[int, int]) -> tuple[np.ndarray, int]:
    if direction == (0, 1):
        patch = np.full((1, length + 1), BG, dtype=int)
        patch[0, 0] = HEAD
        patch[0, 1:] = color
        return patch, 0
    if direction == (0, -1):
        patch = np.full((1, length + 1), BG, dtype=int)
        patch[0, :length] = color
        patch[0, length] = HEAD
        return patch, 0
    if direction == (1, 0):
        patch = np.full((length + 1, 1), BG, dtype=int)
        patch[0, 0] = HEAD
        patch[1:, 0] = color
        return patch, 1

    assert direction == (-1, 0)
    patch = np.full((length + 1, 1), BG, dtype=int)
    patch[:length, 0] = color
    patch[length, 0] = HEAD
    return patch, 0


def _top_row_range(seg: dict, height: int) -> tuple[int, int]:
    length = int(seg["length"])
    direction = seg["direction"]
    if direction == (1, 0):
        return 1, height - length
    if direction == (-1, 0):
        return 0, height - length - 1
    return 0, height - 1


def _assign_group_top_rows(group: list[dict], height: int) -> list[int] | None:
    ranges = [_top_row_range(seg, height) for seg in group]

    def dfs(index: int, prev_top: int, chosen: list[int]) -> list[int] | None:
        if index == len(group):
            return chosen[:]

        lo, hi = ranges[index]
        lo = max(lo, prev_top + 1)
        candidates = list(range(lo, hi + 1))
        random.shuffle(candidates)

        for top in candidates:
            probe_prev = top
            feasible = True
            for j in range(index + 1, len(group)):
                next_lo, next_hi = ranges[j]
                probe_prev = max(next_lo, probe_prev + 1)
                if probe_prev > next_hi:
                    feasible = False
                    break
            if not feasible:
                continue

            chosen.append(top)
            result = dfs(index + 1, top, chosen)
            if result is not None:
                return result
            chosen.pop()
        return None

    return dfs(0, -1, [])


def _assign_top_rows(segments: list[dict], height: int) -> bool:
    lengths = sorted({int(seg["length"]) for seg in segments})
    for length in lengths:
        group = [seg for seg in segments if int(seg["length"]) == length]
        tops = _assign_group_top_rows(group, height)
        if tops is None:
            return False
        for seg, top in zip(group, tops):
            seg["top_row"] = top
    return True


def _pad_window(shape: tuple[int, int], r: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    H, W = shape
    return max(0, r - 1), min(H, r + h + 1), max(0, c - 1), min(W, c + w + 1)


def _can_place(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    r0, r1, c0, c1 = _pad_window(blocked.shape, r, c, h, w)
    return not np.any(blocked[r0:r1, c0:c1])


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    r0, r1, c0, c1 = _pad_window(blocked.shape, r, c, h, w)
    blocked[r0:r1, c0:c1] = True


def _place_scattered_segments(shape: tuple[int, int], start: tuple[int, int], segments: list[dict]) -> np.ndarray | None:
    H, W = shape
    grid = np.full(shape, BG, dtype=int)
    grid[start] = START

    blocked = np.zeros(shape, dtype=bool)
    _mark_blocked(blocked, start[0], start[1], 1, 1)

    placement_order = sorted(
        segments,
        key=lambda seg: np.prod(_segment_patch(seg["color"], seg["length"], seg["direction"])[0].shape),
        reverse=True,
    )

    for seg in placement_order:
        patch, top_rel = _segment_patch(seg["color"], seg["length"], seg["direction"])
        h, w = patch.shape
        r = int(seg["top_row"]) - top_rel
        if not (0 <= r <= H - h):
            return None

        valid_cols = [c for c in range(W - w + 1) if _can_place(blocked, r, c, h, w)]
        if not valid_cols:
            return None

        c = random.choice(valid_cols)
        mask = patch != BG
        region = grid[r : r + h, c : c + w]
        if np.any(region[mask] != BG):
            return None
        region[mask] = patch[mask]
        _mark_blocked(blocked, r, c, h, w)

    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    count = random.randint(*SEGMENT_COUNT_RANGE)
    lengths = _sample_lengths(count)
    colors = random.sample(SEGMENT_COLORS, k=count)
    base_segments = [{"color": colors[i], "length": lengths[i]} for i in range(count)]

    built = _try_build_output(shape, base_segments)
    if built is None:
        return None
    start, ordered_segments, expected = built

    scattered_segments = [dict(seg) for seg in ordered_segments]
    if not _assign_top_rows(scattered_segments, shape[0]):
        return None

    inp = _place_scattered_segments(shape, start, scattered_segments)
    if inp is None:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
